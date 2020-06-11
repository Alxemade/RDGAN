import pickle
import time
from model import *
from utils import *
from config import config, log_config
from scipy.io import loadmat, savemat
import os
import tensorlayer as tl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main_train():
    mask_perc = tl.global_flag['maskperc']
    mask_name = tl.global_flag['mask']
    model_name = tl.global_flag['model']
    mode_lossfunciton = tl.global_flag['lossfunciton']
    # mode_normalization = tl.global_flag['normalization']
    mode_penalty = tl.global_flag['penalty']

    # =================================== BASIC CONFIGS =================================== #

    print('[*] run basic configs ... ')

    log_dir = "logs/log_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(log_dir)
    log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename = logging_setup(log_dir)

    checkpoint_dir = "checkpoints/checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(checkpoint_dir)

    save_dir = "samples/samples_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(save_dir)

    tensorboard_logs = "tensorboard_logs/tensorboard_logs_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(tensorboard_logs)
    # configs
    batch_size = config.TRAIN.batch_size
    early_stopping_num = config.TRAIN.early_stopping_num
    g_alpha = config.TRAIN.g_alpha  # weight for pixel loss
    g_beta = config.TRAIN.g_beta  # weight for frequency loss
    g_gamma = config.TRAIN.g_gamma   # weight for perceptual loss
    g_adv = config.TRAIN.g_adv   # weight for generate network loss
    lr = config.TRAIN.lr  # lr
    lr_decay = config.TRAIN.lr_decay
    decay_every = config.TRAIN.decay_every
    beta1 = config.TRAIN.beta1
    beta2 = config.TRAIN.beta2
    n_epoch = config.TRAIN.n_epoch
    sample_size = config.TRAIN.sample_size
    n_critic = config.TRAIN.n_critic
    log_config(log_all_filename, config)
    log_config(log_eval_filename, config)
    log_config(log_50_filename, config)

    # ==================================== PREPARE DATA ==================================== #

    print('[*] load data ... ')
    training_data_path = config.TRAIN.training_data_path
    val_data_path = config.TRAIN.val_data_path
    testing_data_path = config.TRAIN.testing_data_path

    with open(training_data_path, 'rb') as f:
        X_train = pickle.load(f)

    with open(val_data_path, 'rb') as f:
        X_val = pickle.load(f)

    with open(testing_data_path, 'rb') as f:
        X_test = pickle.load(f)

    print('X_train shape/min/max: ', X_train.shape, X_train.min(), X_train.max())  # (11820, 256, 256, 1) -1.0 1.0
    print('X_val shape/min/max: ', X_val.shape, X_val.min(), X_val.max())  # (3897, 256, 256, 1) -1.0 1.0
    print('X_test shape/min/max: ', X_test.shape, X_test.min(), X_test.max())  # (7914, 256, 256, 1) -1.0 1.0

    print('[*] loading mask ... ')
    if mask_name == "radialcartesi":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_RadialApproxOnCartesi_path, "RadialApproxOnCartesi_{}.mat".format(mask_perc)))[
                'mask']
    elif mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))

    # ==================================== DEFINE MODEL ==================================== #

    print('[*] define model ... ')

    nw, nh, nz = X_train.shape[1:]

    t_image_good = tf.placeholder('float32', [batch_size, nw, nh, nz], name='good_image')
    t_image_good_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='good_image_samples')
    t_image_bad = tf.placeholder('float32', [batch_size, nw, nh, nz], name='bad_image')
    t_image_bad_samples = tf.placeholder('float32', [sample_size, nw, nh, nz], name='bad_image_samples')
    t_gen = tf.placeholder('float32', [batch_size, nw, nh, nz], name='generated_image_for_test')
    t_gen_sample = tf.placeholder('float32', [sample_size, nw, nh, nz], name='generated_sample_image_for_test')
    t_image_good_224 = tf.placeholder('float32', [batch_size, 224, 224, 3], name='vgg_good_image')

    # define generator network
    if tl.global_flag['model'] == 'unet':
        net = u_net_bn(t_image_bad, is_train=True, reuse=False, is_refine=False)
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=True, is_refine=False)
        net_test_sample = u_net_bn(t_image_bad_samples, is_train=False, reuse=True, is_refine=False)

    elif tl.global_flag['model'] == 'unet_refine':
        net = u_net_bn(t_image_bad, is_train=True, reuse=False, is_refine=True)
        net_test = u_net_bn(t_image_bad, is_train=False, reuse=True, is_refine=True)
        net_test_sample = u_net_bn(t_image_bad_samples, is_train=False, reuse=True, is_refine=True)
    else:
        raise Exception("unknown model")

    # calculate generator and discriminator loss
    net_d, logits_fake = discriminator(net.outputs, lossfunciton=mode_lossfunciton,
                                       is_train=True, reuse=False)
    _, logits_real = discriminator(t_image_good, lossfunciton=mode_lossfunciton,
                                   is_train=True, reuse=True)

    # define VGG network
    net_vgg_conv4_good, net_vgg = vgg19_simple_api(t_image_good_224, reuse=False)
    net_vgg_conv4_gen, _ = vgg19_simple_api(tf.tile(tf.image.resize_images(net.outputs, [224, 224]), [1, 1, 1, 3]), reuse=True)

    # ==================================== DEFINE LOSS ==================================== #
    print('[*] define loss functions ... ')
    if mode_lossfunciton == 'gan_loss':
        d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        # discriminator loss (adversarial)
        d_loss = d_loss1 + d_loss2
        # generator loss (adversarial)
        g_loss = tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    elif mode_lossfunciton == 'wgan_loss':
        # discriminator loss (adversarial)
        d_loss = tf.reduce_mean(logits_fake) - tf.reduce_mean(logits_real)
        # generator loss (adversarial)
        g_loss = - tf.reduce_mean(logits_fake)
    elif mode_lossfunciton == 'lsgan_loss':
        d_loss_real = tf.reduce_mean(tf.square(logits_real - 1.0))
        d_loss_fake = tf.reduce_mean(tf.square(logits_fake))
        # discriminator loss (adversarial)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        g_loss = 0.5 * tf.reduce_mean(tf.square(logits_fake - 1.0))
    else:
        raise Exception("Unknow lossfunction")
    # ==================================== DEFINE Penality ==================================== #

    print('[*] define loss functions ... ')
    if mode_penalty == 'no_penalty':
        penalty_loss = 0.0
        d_loss = d_loss + penalty_loss
    elif mode_penalty == 'wgangp_penalty':
        epsilon = tf.random_uniform([], minval=0.0, maxval=1.0)
        x_hat = t_image_good * epsilon + (1 - epsilon) * (net.outputs)
        _, d_hat = discriminator(x_hat, lossfunciton=mode_lossfunciton,
                                 is_train=True, reuse=True)
        gradients = tf.gradients(d_hat, [x_hat])[0]  # calculate the gradients
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 1 * tf.reduce_mean((slopes - 1.0) ** 2)
        d_loss = d_loss + gradient_penalty
    elif mode_penalty == 'dragan_penalty':
        _, var = tf.nn.moments(t_image_good, axes=list(range(len(t_image_good.get_shape()))))
        std = tf.sqrt(var)
        x_noisy = t_image_good + std * (tf.random_uniform(t_image_good.shape) - 0.5)
        x_noisy = tf.clip_by_value(x_noisy, 0.0, 1.0)
        _, d_hat = discriminator(x_noisy, lossfunciton=mode_lossfunciton,
                                 is_train=True, reuse=True)
        gradients = tf.gradients(d_hat, [x_noisy])[0]  # calculate the gradients
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10 * tf.reduce_mean((slopes - 1.0) ** 2)
        d_loss = d_loss + gradient_penalty
    else:
        raise Exception("Unknow penalty")
    # generator loss (perceptual)
    print("=================hee==============")
    print(net_vgg_conv4_good.outputs.shape)
    print(net_vgg_conv4_gen.outputs.shape)
    g_perceptual = tl.cost.mean_squared_error(net_vgg_conv4_good.outputs, net_vgg_conv4_gen.outputs, is_mean=True)

    # generator loss (pixel-wise) 1/2||Xt-Xu(~)||2
    g_nmse_a = tf.sqrt(tf.reduce_sum(tf.squared_difference(net.outputs, t_image_good), axis=[1, 2, 3]))
    g_nmse_b = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    g_nmse = tf.reduce_mean(g_nmse_a / g_nmse_b)

    # generator loss (frequency)
    fft_good_abs = tf.map_fn(fft_abs_for_map_fn, t_image_good)
    fft_gen_abs = tf.map_fn(fft_abs_for_map_fn, net.outputs)
    g_fft = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(fft_good_abs, fft_gen_abs), axis=[1, 2]))

    # generator loss (total)总共损失
    g_loss = g_adv * g_loss + g_alpha * g_nmse + g_gamma * g_perceptual + g_beta * g_fft
    # g_loss = g_adv * g_loss + g_alpha * g_nmse

    nmse_a_0_1 = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen, t_image_good), axis=[1, 2, 3]))
    nmse_b_0_1 = tf.sqrt(tf.reduce_sum(tf.square(t_image_good), axis=[1, 2, 3]))
    nmse_0_1 = nmse_a_0_1 / nmse_b_0_1

    nmse_a_0_1_sample = tf.sqrt(tf.reduce_sum(tf.squared_difference(t_gen_sample, t_image_good_samples), axis=[1, 2, 3]))
    nmse_b_0_1_sample = tf.sqrt(tf.reduce_sum(tf.square(t_image_good_samples), axis=[1, 2, 3]))
    nmse_0_1_sample = nmse_a_0_1_sample / nmse_b_0_1_sample

    # Wasserstein GAN Loss
    with tf.name_scope('WGAN_GP/D'):
        tf.summary.scalar('d_loss', d_loss)

    with tf.name_scope('WGAN_GP/G'):
        tf.summary.scalar('g_loss', g_loss)

    merged = tf.summary.merge_all()
    # ==================================== DEFINE TRAIN OPTS ==================================== #

    print('[*] define training options ... ')

    g_vars = tl.layers.get_variables_with_name('u_net', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1, beta2=beta2).minimize(d_loss, var_list=d_vars)

    # ==================================== TRAINING ==================================== #

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    loss_writer = tf.summary.FileWriter(tensorboard_logs, sess.graph)
    tl.layers.initialize_global_variables(sess)
    sess.run(tf.global_variables_initializer())
    # load generator and discriminator weights (for continuous training purpose)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '_d.npz',
                                 network=net_d)

    net_vgg_conv4_path = config.TRAIN.VGG19_path

    if not os.path.isfile(net_vgg_conv4_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()  # 退出
    npz = np.load(net_vgg_conv4_path, encoding='latin1').item()
    # 这里几点注意
    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)

    net_vgg.print_params(False)
    n_training_examples = len(X_train)
    n_step_epoch = round(n_training_examples / batch_size)

    # sample testing images X_test:7914
    idex = tl.utils.get_random_int(min_v=0, max_v=len(X_test) - 1, number=sample_size, seed=config.TRAIN.seed)
    X_samples_good = X_test[idex]
    # X_samples_good shape: (50,256,256,1)
    X_samples_bad = threading_data(X_samples_good, fn=to_bad_img, mask=mask)

    x_good_sample_rescaled = (X_samples_good + 1) / 2
    x_bad_sample_rescaled = (X_samples_bad + 1) / 2

    tl.visualize.save_images(X_samples_good,
                             [5, 10],
                             os.path.join(save_dir, "sample_image_good.png"))

    tl.visualize.save_images(X_samples_bad,
                             [5, 10],
                             os.path.join(save_dir, "sample_image_bad.png"))

    tl.visualize.save_images(np.abs(X_samples_good - X_samples_bad),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_abs.png"))

    tl.visualize.save_images(np.sqrt(np.abs(X_samples_good - X_samples_bad) / 2 + config.TRAIN.epsilon),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_sqrt_abs.png"))

    tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - X_samples_bad) / 2, 0, 1),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_diff_sqrt_abs_10_clip.png"))

    tl.visualize.save_images(threading_data(X_samples_good, fn=distort_img),
                             [5, 10],
                             os.path.join(save_dir, "sample_image_aug.png"))
    scipy.misc.imsave(os.path.join(save_dir, "mask.png"), mask * 255)

    print('[*] start training ... ')

    best_nmse = np.inf
    best_epoch = 1
    esn = early_stopping_num
    for epoch in range(0, n_epoch):

        # learning rate decay
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            log_all.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
            log_all.debug(log)

        for step in range(n_step_epoch):
            step_time = time.time()
            # now train the generator once!
            idex = tl.utils.get_random_int(min_v=0, max_v=n_training_examples - 1, number=batch_size)
            X_good = X_train[idex]
            X_good_aug = threading_data(X_good, fn=distort_img)
            X_good_224 = threading_data(X_good_aug, fn=vgg_prepro)
            X_bad = threading_data(X_good_aug, fn=to_bad_img, mask=mask)
            sess.run(d_optim, {t_image_good: X_good_aug, t_image_bad: X_bad})
            sess.run(g_optim, {t_image_good_224: X_good_224, t_image_good: X_good_aug, t_image_bad: X_bad})

            errG, errG_perceptual, errG_nmse, errG_fft, errD, summary = sess.run([g_loss, g_perceptual, g_nmse, g_fft, d_loss, merged],
                                                                                 {t_image_good_224: X_good_224,
                                                                                  t_image_good: X_good_aug,
                                                                                  t_image_bad: X_bad})
            loss_writer.add_summary(summary, n_epoch)


            log = "Epoch[{:3}/{:3}] step={:3} d_loss={:5} g_loss={:5} g_perceptual_loss={:5} g_mse={:5} g_freq={:5} took {:3}s".format(
                epoch + 1,
                n_epoch,
                step,
                round(float(errD), 3),
                round(float(errG), 3),
                round(float(errG_perceptual), 3),
                round(float(errG_nmse), 3),
                round(float(errG_fft), 3),
                round(time.time() - step_time, 2))

            print(log)
            log_all.debug(log)

        # evaluation for training data
        total_nmse_training = 0
        total_ssim_training = 0
        total_psnr_training = 0
        num_training_temp = 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=X_train, batch_size=batch_size, shuffle=False):
            x_good, _ = batch
            # x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
            x_bad = threading_data(
                x_good,
                fn=to_bad_img,
                mask=mask)

            x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

            x_good_0_1 = (x_good + 1) / 2
            x_gen_0_1 = (x_gen + 1) / 2

            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
            ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
            total_nmse_training += np.sum(nmse_res)
            total_ssim_training += np.sum(ssim_res)
            total_psnr_training += np.sum(psnr_res)
            num_training_temp += batch_size

        total_nmse_training /= num_training_temp
        total_ssim_training /= num_training_temp
        total_psnr_training /= num_training_temp

        log = "Epoch: {}\nNMSE training: {:8}, SSIM training: {:8}, PSNR training: {:8}".format(
            epoch + 1,
            total_nmse_training,
            total_ssim_training,
            total_psnr_training)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        # evaluation for validation data
        total_nmse_val = 0
        total_ssim_val = 0
        total_psnr_val = 0
        num_val_temp = 0
        for batch in tl.iterate.minibatches(inputs=X_val, targets=X_val, batch_size=batch_size, shuffle=False):
            x_good, _ = batch
            # x_bad = threading_data(x_good, fn=to_bad_img, mask=mask)
            x_bad = threading_data(
                x_good,
                fn=to_bad_img,
                mask=mask)

            x_gen = sess.run(net_test.outputs, {t_image_bad: x_bad})

            x_good_0_1 = (x_good + 1) / 2
            x_gen_0_1 = (x_gen + 1) / 2

            nmse_res = sess.run(nmse_0_1, {t_gen: x_gen_0_1, t_image_good: x_good_0_1})
            ssim_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_0_1, x_gen_0_1)], fn=psnr)
            total_nmse_val += np.sum(nmse_res)
            total_ssim_val += np.sum(ssim_res)
            total_psnr_val += np.sum(psnr_res)
            num_val_temp += batch_size

        total_nmse_val /= num_val_temp
        total_ssim_val /= num_val_temp
        total_psnr_val /= num_val_temp

        log = "Epoch: {}\nNMSE val: {:8}, SSIM val: {:8}, PSNR val: {:8}".format(
            epoch + 1,
            total_nmse_val,
            total_ssim_val,
            total_psnr_val)
        print(log)
        log_all.debug(log)
        log_eval.info(log)

        img = sess.run(net_test_sample.outputs, {t_image_bad_samples: X_samples_bad})
        tl.visualize.save_images(img,
                                 [5, 10],
                                 os.path.join(save_dir, "image_{}.png".format(epoch)))

        if total_nmse_val < best_nmse:
            esn = early_stopping_num
            best_nmse = total_nmse_val
            best_epoch = epoch + 1

            # save current best model
            tl.files.save_npz(net.all_params,
                              name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                              sess=sess)

            tl.files.save_npz(net_d.all_params,
                              name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '_d.npz',
                              sess=sess)
            print("[*] Save checkpoints SUCCESS!")
        else:
            esn -= 1

        log = "Best NMSE result: {} at {} epoch".format(best_nmse, best_epoch)
        log_eval.info(log)
        log_all.debug(log)
        print(log)

        if esn == 0:
            log_eval.info(log)

            tl.files.load_and_assign_npz(sess=sess,
                                         name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                         network=net)
            # evluation for test data
            x_gen = sess.run(net_test_sample.outputs, {t_image_bad_samples: X_samples_bad})
            x_gen_0_1 = (x_gen + 1) / 2
            savemat(save_dir + '/test_random_50_generated.mat', {'x_gen_0_1': x_gen_0_1})

            nmse_res = sess.run(nmse_0_1_sample, {t_gen_sample: x_gen_0_1, t_image_good_samples: x_good_sample_rescaled})
            ssim_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=ssim)
            psnr_res = threading_data([_ for _ in zip(x_good_sample_rescaled, x_gen_0_1)], fn=psnr)

            log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
                nmse_res,
                ssim_res,
                psnr_res)

            log_50.debug(log)

            log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
                np.mean(nmse_res),
                np.mean(ssim_res),
                np.mean(psnr_res))

            log_50.debug(log)

            log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
                                                                                                np.std(ssim_res),
                                                                                                np.std(psnr_res))

            log_50.debug(log)

            # evaluation for zero-filled (ZF) data,
            nmse_res_zf = sess.run(nmse_0_1_sample,
                                   {t_gen_sample: x_bad_sample_rescaled, t_image_good_samples: x_good_sample_rescaled})
            ssim_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=ssim)
            psnr_res_zf = threading_data([_ for _ in zip(x_good_sample_rescaled, x_bad_sample_rescaled)], fn=psnr)

            log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
                nmse_res_zf,
                ssim_res_zf,
                psnr_res_zf)

            log_50.debug(log)

            log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
                np.mean(nmse_res_zf),
                np.mean(ssim_res_zf),
                np.mean(psnr_res_zf))

            log_50.debug(log)

            log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
                np.std(nmse_res_zf),
                np.std(ssim_res_zf),
                np.std(psnr_res_zf))

            log_50.debug(log)

            # sample testing images
            tl.visualize.save_images(x_gen,
                                     [5, 10],
                                     os.path.join(save_dir, "final_generated_image.png"))

            tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - x_gen) / 2, 0, 1),
                                     [5, 10],
                                     os.path.join(save_dir, "final_generated_image_diff_abs_10_clip.png"))


            tl.visualize.save_images(np.clip(10 * np.abs(X_samples_good - X_samples_bad) / 2, 0, 1),
                                     [5, 10],
                                     os.path.join(save_dir, "final_bad_image_diff_abs_10_clip.png"))

            print("[*] Job finished!")
            break

def main_evaluate():
    # load mask
    mask_perc = tl.global_flag['maskperc']
    mask_name = tl.global_flag['mask']
    model_name = tl.global_flag['model']
    # load the parameters

    log_dir = "logs/log_{}_{}_{}".format(model_name, mask_name, mask_perc)
    tl.files.exists_or_mkdir(log_dir)
    log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename = logging_setup(log_dir)
    log_config(log_50_filename, config)
    checkpoint_dir = "checkpoints/checkpoint_{}_{}_{}".format(model_name, mask_name, mask_perc)
    # samples the save of the evaluation
    evaluate_data_sample_path = config.TRAIN.evaluate_data_sample_path
    tl.files.exists_or_mkdir(evaluate_data_sample_path)
    evaluate_data_save_path = config.TRAIN.evaluate_data_save_path
    tl.files.exists_or_mkdir(evaluate_data_save_path)

    print('[*] loading mask ... ')
    if mask_name == "radialcartesi":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_RadialApproxOnCartesi_path,
                             "RadialApproxOnCartesi_{}.mat".format(mask_perc)))[
                'mask']
    elif mask_name == "gaussian2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian2D_path, "GaussianDistribution2DMask_{}.mat".format(mask_perc)))[
                'maskRS2']
    elif mask_name == "gaussian1d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "GaussianDistribution1DMask_{}.mat".format(mask_perc)))[
                'maskRS1']
    elif mask_name == "poisson2d":
        mask = \
            loadmat(
                os.path.join(config.TRAIN.mask_Gaussian1D_path, "PoissonDistributionMask_{}.mat".format(mask_perc)))[
                'population_matrix']
    else:
        raise ValueError("no such mask exists: {}".format(mask_name))

    print('[*] load evaluate data ... ')
    evaluate_gd_img_list = sorted(
        tl.files.load_file_list(path=evaluate_data_sample_path, regx='.*.png', printable=False))
    evaluate_gd_imgs = tl.vis.read_images(evaluate_gd_img_list, path=evaluate_data_sample_path, n_threads=1)
    ###========================== DEFINE MODEL ============================###
    # imid = 0  # the first image
    # evaluate_gd_img = evaluate_gd_imgs[imid]
    print("=================test===============")
    test_length = len(evaluate_gd_imgs)
    test_good_image = []
    for i in range(test_length):
        evaluate_gd_img = evaluate_gd_imgs[i]
        evaluate_gd_img = evaluate_gd_img[:, :, np.newaxis]
        # print(evaluate_gd_img.shape)
        evaluate_gd_img = evaluate_gd_img / 127.5 - 1
        test_good_image.append(evaluate_gd_img)

    # print("lens of test", len(test_list))

    evaluate_samples_bad = threading_data(test_good_image, fn=to_bad_img, mask=mask)

    x_bad_sample_rescaled = (evaluate_samples_bad + 1) / 2
    print("====================test length==================")
    size = evaluate_samples_bad.shape
    print(size)
    print(len(evaluate_samples_bad))
    evaluate_image = tf.placeholder('float32', [None, size[1], size[2], size[3]], name='input_image')
    t_gen_sample = tf.placeholder('float32', [None, size[1], size[2], size[3]], name='generated_sample_image_for_test')
    t_image_good_samples = tf.placeholder('float32', [None, size[1], size[2], size[3]], name='good_image_samples')
    nmse_a_0_1_sample = tf.sqrt(
        tf.reduce_sum(tf.squared_difference(t_gen_sample, t_image_good_samples), axis=[1, 2, 3]))
    nmse_b_0_1_sample = tf.sqrt(tf.reduce_sum(tf.square(t_image_good_samples), axis=[1, 2, 3]))
    nmse_0_1_sample = nmse_a_0_1_sample / nmse_b_0_1_sample
    ###========================== RESTORE G =============================###
    # define generator network
    if tl.global_flag['model'] == 'unet':
        net = u_net_bn(evaluate_image, is_train=False, reuse=False, is_refine=False)

    elif tl.global_flag['model'] == 'unet_refine':
        net = u_net_bn(evaluate_image, is_train=False, reuse=False, is_refine=True)

    else:
        raise Exception("unknown model")

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, tl.global_flag['model']) + '.npz',
                                 network=net)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    evaluate_restore_img = sess.run(net.outputs, {evaluate_image: evaluate_samples_bad})
    print("took: %4.4fs" % (time.time() - start_time))
    print("zero filled size : %s / restore image size: %s" % (size, evaluate_restore_img.shape))

    for i in range(test_length):
        if i >= 0 and i < 9:
            # tl.vis.save_image(evaluate_restore_img[i], evaluate_data_save_path + '/resore/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/resore/00{}.mat'.format(i+1), {'dagangp': evaluate_restore_img[i]})
            # tl.vis.save_image(evaluate_samples_bad[i], evaluate_data_save_path + '/zerofilled/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/zerofilled/00{}.mat'.format(i + 1), {'zerofill': evaluate_samples_bad[i]})
        elif i >= 9 and i < 99:
            # tl.vis.save_image(evaluate_restore_img[i], evaluate_data_save_path + '/resore/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/resore/0{}.mat'.format(i + 1), {'dagangp': evaluate_restore_img[i]})
            # tl.vis.save_image(evaluate_samples_bad[i], evaluate_data_save_path + '/zerofilled/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/zerofilled/0{}.mat'.format(i + 1),{'zerofill': evaluate_samples_bad[i]})
        else:
            # tl.vis.save_image(evaluate_restore_img[i], evaluate_data_save_path + '/resore/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/resore/{}.mat'.format(i + 1), {'dagangp': evaluate_restore_img[i]})
            # tl.vis.save_image(evaluate_samples_bad[i], evaluate_data_save_path + '/zerofilled/00{}.png'.format(i + 1))
            savemat(evaluate_data_save_path + '/zerofilled/{}.mat'.format(i + 1),{'zerofill': evaluate_samples_bad[i]})
    # calculatge the NMSE, ssim, psnr
    evaluate_restore_img_rescaled = (evaluate_restore_img + 1) / 2
    evaluate_gd_img_rescaled = []
    for i in range(test_length):
        test_good_image[i] = (test_good_image[i] + 1) / 2
        evaluate_gd_img_rescaled.append(test_good_image[i])

    # print("evaluate_restore_img_rescaled shape")
    # print(evaluate_restore_img_rescaled.shape)
    # print(evaluate_gd_img_rescaled.shape)
    print("start evaluate ssim and PSNR")
    print("===================start: good-restore================")
    ssim_res = threading_data([_ for _ in zip(evaluate_gd_img_rescaled, evaluate_restore_img_rescaled)], fn=ssim)
    psnr_res = threading_data([_ for _ in zip(evaluate_gd_img_rescaled, evaluate_restore_img_rescaled)], fn=psnr)
    nmse_res = sess.run(nmse_0_1_sample, {t_gen_sample: evaluate_restore_img_rescaled, t_image_good_samples: evaluate_gd_img_rescaled})

    log = "NMSE testing: {}\nSSIM testing: {}\nPSNR testing: {}\n\n".format(
        nmse_res,
        ssim_res,
        psnr_res)

    log_50.debug(log)

    log = "NMSE testing average: {}\nSSIM testing average: {}\nPSNR testing average: {}\n\n".format(
        np.mean(nmse_res),
        np.mean(ssim_res),
        np.mean(psnr_res))

    log_50.debug(log)

    log = "NMSE testing std: {}\nSSIM testing std: {}\nPSNR testing std: {}\n\n".format(np.std(nmse_res),
                                                                                        np.std(ssim_res),
                                                                                        np.std(psnr_res))
    #
    log_50.debug(log)
    print("===================start: good-zf ================")
    ssim_res_zf = threading_data([_ for _ in zip(evaluate_gd_img_rescaled, x_bad_sample_rescaled)], fn=ssim)
    psnr_res_zf = threading_data([_ for _ in zip(evaluate_gd_img_rescaled, x_bad_sample_rescaled)], fn=psnr)
    nmse_res_zf = sess.run(nmse_0_1_sample,
                           {t_gen_sample: x_bad_sample_rescaled, t_image_good_samples: evaluate_gd_img_rescaled})

    log = "NMSE ZF testing: {}\nSSIM ZF testing: {}\nPSNR ZF testing: {}\n\n".format(
        nmse_res_zf,
        ssim_res_zf,
        psnr_res_zf)

    log_50.debug(log)

    log = "NMSE ZF average testing: {}\nSSIM ZF average testing: {}\nPSNR ZF average testing: {}\n\n".format(
        np.mean(nmse_res_zf),
        np.mean(ssim_res_zf),
        np.mean(psnr_res_zf))

    log_50.debug(log)

    log = "NMSE ZF std testing: {}\nSSIM ZF std testing: {}\nPSNR ZF std testing: {}\n\n".format(
        np.std(nmse_res_zf),
        np.std(ssim_res_zf),
        np.std(psnr_res_zf))

    log_50.debug(log)
    print("job finished")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='unet', help='unet,unet_refine')
    parser.add_argument('--mask', type=str, default='radialcartesi', help='radialcartesi, gaussian1d, gaussian2d, poisson2d')
    parser.add_argument('--maskperc', type=int, default='30', help='10,20,30,40,50')
    parser.add_argument('--lossfunciton', type=str, default='gan_loss',
                        help='gan_loss, wgan_loss, lsgan_loss')
    # parser.add_argument('--normalization', type=str, default='layer_norm',
    #                     help='no_normalization, layer_norm, spectral_norm')
    parser.add_argument('--penalty', type=str, default='wgangp_penalty',
                        help='no_penalty, wgangp_penalty, dragan_penalty')
    parser.add_argument('--train', type=str, default='train', help='train,evaluate')
    args = parser.parse_args()

    tl.global_flag['model'] = args.model
    tl.global_flag['mask'] = args.mask
    tl.global_flag['maskperc'] = args.maskperc
    tl.global_flag['lossfunciton'] = args.lossfunciton
    # tl.global_flag['normalization'] = args.normalization
    tl.global_flag['penalty'] = args.penalty
    tl.global_flag['train'] = args.train
    if tl.global_flag['train'] == 'train':
        main_train()
    elif tl.global_flag['train'] == 'evaluate':
        main_evaluate()
    else:
        raise Exception("Unknow --train")
