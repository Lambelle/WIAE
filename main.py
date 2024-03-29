import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
import utils
import argparse
from statistical_tests import runs_up_and_down
from utils import empirical_cdf_transform, reshape_innovations, ROC_curve_plotting
from tabulate import tabulate




class WGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        decoder,
        de_discriminator,
        discriminator_extra_steps=3,
        gp_weight=10.0,
        gp_d=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.decoder = decoder
        self.ded = de_discriminator
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.gp_d = gp_d

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_samples, fake_samples, IsDeD=False):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interplated image
        alpha = tf.random.uniform(shape=[batch_size, 1], minval=0.0, maxval=1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            if not IsDeD:
                pred = self.discriminator(interpolated, training=True)
            else:
                pred = self.ded(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-10)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # Get the batch size
        batch_size = tf.shape(data)[0]
        for i in range(self.d_steps):
            # Get the latent vector
            z = data
            z_ded = tf.cast(z[:, -(nI - 2 * (k_size - 1)) :], tf.float32)
            real_samples = tf.cast(
                tf.random.uniform(
                    (batch_size, nI - k_size + 1), minval=-1.0, maxval=1.0
                ),
                tf.float32,
            )
            with tf.GradientTape(persistent=True) as tape:
                # Generate fake images from the latent vector
                fake_samples = self.generator(z, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_samples, training=True)
                # Get the logits for real images
                real_logits = self.discriminator(real_samples, training=True)
                # Get samples from decoder
                decoded_samples = self.decoder(fake_samples)
                decoded_samples_ded = decoded_samples[:, -(nI - 2 * (k_size - 1)) :]
                # Calculate discriminator loss using fake and real logits
                d_cost = self.d_loss_fn(
                    real_sample=real_logits, fake_sample=fake_logits
                )
                # Calculate autoencoder discriminator logits
                real_ded_logits = self.ded(z_ded)
                fake_ded_logits = self.ded(decoded_samples_ded)
                # Calculate autoencoder discriminator loss
                ded_cost = self.d_loss_fn(
                    real_sample=real_ded_logits, fake_sample=fake_ded_logits
                )
                # Calculate the gradient penalty
                gp = self.gradient_penalty(
                    batch_size, real_samples, fake_samples, IsDeD=False
                )
                # Calculate the gradient penalty for autoencoder
                gp_ded = self.gradient_penalty(
                    batch_size, z_ded, decoded_samples_ded, IsDeD=True
                )
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight
                # Add the gradient penalty to the original discriminator loss
                ded_loss = ded_cost + gp_ded * self.gp_d

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            # Stochasctic Gradient descent for autoencoder discriminator
            ded_gradient = tape.gradient(ded_loss, self.ded.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(ded_gradient, self.ded.trainable_variables)
            )

        # Train the generator now.
        # Get the latent vector
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images using the generato
            generated_images = self.generator(z, training=True)
            # Get output from decoder
            decoded_images = self.decoder(generated_images)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Get the autoencoder discriminator logits for decoder samples
            decoded_images_de = decoded_images[:, -(nI - 2 * (k_size - 1)) :]
            decoded_logits = self.ded(decoded_images_de)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits, decoded_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        # Get the gradients w.r.t the decoder loss
        dec_gradient = tape.gradient(g_loss, self.decoder.trainable_variables)
        # Update the weights of the decoder using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(dec_gradient, self.decoder.trainable_variables)
        )
        return {
            "Independence Discriminator Loss": d_loss,
            "Encoder Loss": g_loss,
            "Equivalent Discriminator Loss": ded_loss,
        }

    def call(self, input):
        output = self.generator(input, training=False)
        return output


class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    # def on_epoch_end(self, epoch, logs=None):
    #     z = x_train
        # output = self.model.generator(z)
        # output = output.numpy()
        # b = 50
        # output_c = np.empty((output.shape[0] - b - 1, b))
        # tf.print("Prediction at epochs:".format(output, epoch))
        # bins = np.arange(0, 2, 0.001) - 1
        # plt.hist(output)
        # plt.show()


def discriminator_loss(real_sample, fake_sample):
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    return fake_loss - real_loss


def generator_loss(fake_sample, decoded_logits):
    return -tf.reduce_mean(fake_sample) - de_w * tf.reduce_mean(decoded_logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-data", required=True)
    parser.add_argument("-data_bad", required=True)
    parser.add_argument("-test_perc", type=float, default=0.2)

    parser.add_argument("-degree", type=int, default=4)
    parser.add_argument("-block", type=int, default=100)
    parser.add_argument("-stride", type=int, default=100)

    parser.add_argument("-nO", type=int, default=1)
    parser.add_argument("-nD", type=int, default=100)
    parser.add_argument("-nI", type=int, default=50)
    parser.add_argument("-k_size", type=int, default=20)

    parser.add_argument("-batchsize", type=int, default=60)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lrD", type=float, default=1e-3)
    parser.add_argument("-lrG", type=float, default=1e-3)
    parser.add_argument("-trainingC", type=int, default=1e-3)

    parser.add_argument("-gp_W", type=float, default=1)
    parser.add_argument("-de_W", type=float, default=1)
    parser.add_argument("-gp_W_ded", type=float, default=1)

    parser.add_argument("-seed", type=int, default=1)

    opt = parser.parse_args()


    nO = opt.nO
    nD = opt.nD
    k_size = opt.k_size
    nI = opt.nI

    lrD = opt.lrD
    lrG = opt.lrG
    batchsize = opt.batchsize
    epochs = opt.epochs

    trainingC = opt.trainingC
    gp_w = opt.gp_W
    de_w = opt.de_W  # Weight for decoder
    gp_w_ded = opt.gp_W_ded

    ts_perc = opt.test_perc
    data = opt.data
    data_bad = opt.data_bad

    fname_data = "dataset/" + data
    fname_data_bad = "dataset/" + opt.data_bad

    degree = opt.degree
    block_size = opt.block
    strides = opt.stride

    seed_number = opt.seed

    tf.random.set_seed(seed_number)

    dataSet = np.loadtxt(fname_data, delimiter=",")
    dataSize = dataSet.size
    tr_size = int(dataSize * (1 - ts_perc))
    ts_size = int(dataSize * ts_perc)

    train_samples = dataSet[0 : tr_size - 1]
    test_samples = dataSet[tr_size : tr_size + ts_size - 1]
    bad_samples = np.loadtxt(fname_data_bad, delimiter=",")

    print("train_samples:", train_samples.shape)
    print("test_samples:", test_samples.shape)
    print("bad samples:", bad_samples.shape)

    x_train = utils.get_training_samples(train_samples, nI, k_size)
    x_bad = utils.get_training_samples(bad_samples, nI, k_size)
    x_test = utils.get_training_samples(test_samples, nI, k_size)

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lrG, beta_1=0.9, beta_2=0.999
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=lrD, beta_1=0.9, beta_2=0.999
    )

    D_init = tf.keras.initializers.he_normal(seed=1)
    D = model.get_discriminator_model(D_init, nD, nO, nI, k_size)
    # D.summary()

    G_init = tf.keras.initializers.he_normal(seed=1)
    G = model.get_generator_model(G_init, nD, nI, k_size)
    # G.summary()

    Decoder_init = tf.keras.initializers.he_normal(seed=0)
    De = model.get_decoder_model(Decoder_init, nI, k_size, nD)
    # De.summary()

    Ded_init = tf.keras.initializers.he_normal(seed=0)
    DeD = model.get_discriminator_de_model(Ded_init, nI, nD, nO, k_size)
    # DeD.summary()

    cbk = GANMonitor()

    wgan = WGAN(
        discriminator=D,
        generator=G,
        decoder=De,
        de_discriminator=DeD,
        discriminator_extra_steps=trainingC,
        gp_weight=gp_w,
        gp_d=gp_w_ded,
    )
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )
    wgan.fit(
        x_train,
        batch_size=batchsize,
        epochs=epochs,
        callbacks=[cbk],
    )

    inn_train = wgan.generator(x_train).numpy()
    inn_bad = wgan.generator(x_bad).numpy()
    inn_test = wgan.generator(x_test).numpy()

    recons_train = wgan.decoder(inn_train).numpy()
    recons_bad = wgan.decoder(inn_bad).numpy()
    recons_test = wgan.decoder(inn_test).numpy()

    inn_train = inn_train.flatten()
    inn_bad = inn_bad.flatten()
    inn_test = inn_test.flatten()

    fname_recon_test = "results/recons_" + data
    fname_recon_bad = "results/recons_" + data

    np.savetxt(fname_recon_test, recons_test)
    np.savetxt(fname_recon_bad, recons_bad)

    z_ud_bad = runs_up_and_down(inn_bad)
    z_ud_test = runs_up_and_down(inn_test)

    print_data = [["test", z_ud_test], ["bad", z_ud_bad]]
    print(tabulate(print_data, headers=["data", "p_value for runs test"]))

    inn_test_transformed = empirical_cdf_transform(inn_train, inn_test)
    inn_bad_transformed = empirical_cdf_transform(inn_train, inn_bad)

    fname_inn_test = "results/inn_" + data
    fname_inn_bad = "results/inn_" + data

    dataname = data.replace("txt", "")
    fig_name = (
        "ROC_curves/lrG_"
        + str(lrG)
        + "lrD_"
        + str(lrD)
        + "gp_w_"
        + str(gp_w)
        + "gp_w_ded_"
        + str(gp_w_ded)
        + "de_w_"
        + str(de_w)
        + "seed_"
        + str(seed_number)
        + "_runs_test_"
        + str(z_ud_test)
        + "_"
        + dataname
        + ".png"
    )

    # plt.hist(inn_test_transformed.flatten())
    # plt.show()
    # plt.hist(inn_bad_transformed.flatten())
    # plt.show()

    np.savetxt(fname_inn_test, inn_test_transformed)
    np.savetxt(fname_inn_bad, inn_bad_transformed)

    inn_test_transformed_reshape = reshape_innovations(
        inn_test_transformed, block_size, strides
    )
    inn_bad_transformed_reshape = reshape_innovations(
        inn_bad_transformed, block_size, strides
    )

    true_positive, false_positive = ROC_curve_plotting(
        inn_test_transformed_reshape, inn_bad_transformed_reshape, degree, fig_name
    )
    fname_roc_TP = "ROC_curves/TP_" + data
    fname_roc_FP = "ROC_curves/FP_" + data

    np.savetxt(fname_roc_TP, true_positive)
    np.savetxt(fname_roc_FP, false_positive)
