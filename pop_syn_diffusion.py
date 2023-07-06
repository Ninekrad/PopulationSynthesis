
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

if numerical_col_n > 0:
    scaler = MinMaxScaler()
    scaler = scaler.fit(data_train[:, :numerical_col_n])
    data_train[:, :numerical_col_n] = scaler.transform(data_train[:, :numerical_col_n])

i = 0
temp_data_train = []
temp_data_train.append(np.concatenate([data_train[i], np.zeros(256-136)]))

total_data_train = []
for i in range(data_train.shape[0]):
    total_data_train.append(np.concatenate([data_train[i], np.zeros(256-136)]))

total_data_train = np.array(total_data_train)
total_data_train = total_data_train.reshape(data_train.shape[0],16,16,1)

"""copy all data to make 3 layers"""

total_data_train_copied = []
for i in range(data_train.shape[0]):
    total_data_train_copied.append(np.concatenate([total_data_train[i], total_data_train[i], total_data_train[i]]))

# total_data_train_copied[0]

total_data_train = np.array(total_data_train_copied)

data_train_fil = total_data_train.reshape(data_train.shape[0],3,16,16)
data_train_fil = data_train_fil.transpose(0,2,3,1).reshape(-1,16,16,3)
print(data_train_fil[0].shape)

train_data = data_train_fil[:np.int(data_train.shape[0]*0.8)]
val_data = data_train_fil[np.int(data_train.shape[0]*0.8):]

print(train_data.shape)
print(val_data.shape)

train_data_T = tf.data.Dataset.from_tensor_slices(train_data).cache().repeat(dataset_repetitions).shuffle(10 * batch_size).batch(batch_size, drop_remainder=True)
val_data_T = tf.data.Dataset.from_tensor_slices(val_data).cache().repeat(dataset_repetitions).shuffle(10 * batch_size).batch(batch_size, drop_remainder=True)
print(train_data_T)
print(val_data_T)

class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()

def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")

class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

    def generate_samples(self, epoch=None, logs=None, num_samples=10000):
        # plot random generated images for visual evaluation of generation quality
        generated_data = self.generate(
            num_images=num_samples,
            diffusion_steps=plot_diffusion_steps,
        )

        generated_sample_list =[]
        
        for index in range(num_samples):
            temp_generated_data = generated_data[index]
            generated_sample_list.append(temp_generated_data[:,:,0])
            generated_sample_list.append(temp_generated_data[:,:,1])
            generated_sample_list.append(temp_generated_data[:,:,2])
            #generated_sample_list.append(generated_data)

        return generated_sample_list

# create and compile the model
model = DiffusionModel(image_size, widths, block_depth)

print("Compile Start")
model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)
# pixelwise mean absolute error is used as loss

# save the best model based on the validation KID metric
# checkpoint_path = "/app/outputs/"

print("Check")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=args.checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,
)

# calculate mean and variance of training dataset for normalization

print("Model Normalk")
model.normalizer.adapt(train_data_T)
model.normalizer.adapt(val_data_T)

"""Diffusion model fitting"""

# run training and plot generated images periodically
model.fit(
    train_data_T,
    epochs=num_epochs,
    validation_data=val_data_T,
    callbacks=[
        #keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images),
        checkpoint_callback,
    ],
    verbose=2,
)

model.plot_images()

lll = model.generate_samples()

t = np.array(lll[0])
tt = t.reshape(256,-1)
ttt = tt[:136]
print(len(ttt))

diffusion_samples = []
for i in range(30000):
    tt = np.array(lll[i])
    tt = tt.reshape(256,-1)
    ttt = tt[:136,0]
    diffusion_samples.append(ttt.tolist())

diffusion_samples = np.array(diffusion_samples)

#get samples from Diffusion model

def diffusion_get_samples(diffusion_samples, n_samples, scaler):
    #samples = model.generate_samples()
    samples = diffusion_samples
    # scale back
    if numerical_col_n > 0:
        samples[:,:numerical_col_n] = scaler.inverse_transform(samples[:,:numerical_col_n])
    # back from categorical softmax to one-hot
    for g_i in range(cat_groups_n):
        g_i_beg = cat_groups[g_i]
        g_i_end = cat_groups[g_i + 1]
        data_pred_col = samples[:, g_i_beg:g_i_end]
        data_pred_col = np.argmax(data_pred_col, axis=1)
        for row_i, row in enumerate(samples):
            for col_i in range(g_i_beg, g_i_end):
                if col_i - g_i_beg != data_pred_col[row_i]:
                    samples[row_i, col_i] = 0
                else:
                    samples[row_i, col_i] = 1
    # deal with the integer variables and clip using min/max values
    if numerical_col_n > 0:
        min_max_scheme = 'scheme_1'
        col_names_num = df.columns.tolist()[:numerical_col_n]
        for col_ind, col_name in enumerate(col_names_num):
            if col_name in numerical_int:
                samples[:, col_ind] = np.around(samples[:, col_ind])
            # 
            samples[:, col_ind] = np.clip(samples[:, col_ind], 
                                    min_max_bins[min_max_scheme][col_name][0], 
                                    min_max_bins[min_max_scheme][col_name][1])
    return samples

samples_diffusion = diffusion_get_samples(diffusion_samples, 30000, scaler)

def check_marginals_numerical2(data, num_bin, methodType):
    print('\n--- NUMERICAL ---')
    col_names_num = df.columns.tolist()[:numerical_col_n]
    for col_ind, col_name in enumerate(col_names_num):
        print(80*'-')
        print(col_name)
        d_cols = []
        for d in data:
            d_col = d[:, col_ind]
            if col_name in numerical_int:
                d_col = np.around(d_col)
                print("//int//")
            print(d_col)
            print(num_bin)
            print(col_name)
            print(min_max_bins)
            print(num_bin)
            print(min_max_bins[num_bin[0]][col_name])
            print(min_max_bins[num_bin[0]][col_name][0])
            d_col = np.clip(d_col, 
                            min_max_bins[num_bin[0]][col_name][0], 
                            min_max_bins[num_bin[0]][col_name][1])
            d_cols.append(d_col)
            print(d_col[:10])
        title = col_name.split('--')[0] + "_" + methodType
        check_marginals(d_cols, bins=min_max_bins[num_bin[0]][col_name][2], title=title)
        print(80*'-')

print(80 * '-')
check_marginals_numerical2([data_train, samples_diffusion], bin_n_comparisons, "DIFFUSION")
print(80 * '-')

print(80 * '-')
check_marginals_categorical([data_train, samples_diffusion], "DIFFUSION")
print(80 * '-')

samples_diffusion_dic = {"scheme_1": samples_diffusion}

full_diffusion_model_cache_dir = os.path.join(model_cache_dir, 'full_diffusion')
stat_diffusion = get_stat(data_test, samples_diffusion_dic, data_train, bin_n_comparisons,
                     samples_true_cache_dir, full_diffusion_model_cache_dir)

bin_ns_diffusion = ['scheme_1']

# performance for all atributes
(errors_diffusion, 
 diversity_1_diffusion, 
 diversity_2_diffusion,
 xticks_diffusion) = get_perform_plot_data(stat_diffusion, bin_n_comparisons, bin_ns_diffusion)