# =====================================================================================
# GENERAL PURPOSE LIBRARIES
# =====================================================================================

import numpy as np, tensorflow as tf, os, glob
from tensorflow.keras import Input, Model, layers, losses, optimizers, callbacks

# =====================================================================================
# CUSTOM LIBRARIES DEVELOPED AND MAINTAINED BY DR. PETER CHANG
# =====================================================================================

from jarvis.train.client import Client
from jarvis.train import params, custom
from jarvis.utils.general import gpus, overload, tools as jtools
from jarvis.utils.display import montage
from jarvis.tools import show

# =====================================================================================
# ALL SOURCE CODE BELOW WAS DEVELOPED AND MAINTAINED BY SU KARA
# =====================================================================================

# =====================================================================================
# DEFINE CLIENT 
# =====================================================================================

@overload(Client)
def preprocess(self, arrays, **kwargs):
    """
    Method to preprocess arrays

    At the beginning of this method:

      arrays['xs']['dat'] ==> full 512 x 512 mammogram
      arrays['ys']['lbl'] ==> breast + fiboglandular mask

    By the end of this method:

      arrays['xs']['dat'] ==> cropped N x N patches
      arrays['ys']['lbl'] ==> target reconstruction image scaled between [0, 1] 

    """
    # --- Create 8 crops from fg
    fg_crops = self.create_random_crops(arrays, N=8, mask=arrays['ys']['lbl'] == 2)

    # --- Create 8 crops from bg
    bg_crops = self.create_random_crops(arrays, N=8, mask=arrays['ys']['lbl'] == 1)

    # --- Assemble dat
    dat = np.concatenate((
        fg_crops['xs']['dat'],
        bg_crops['xs']['dat'])) 

    # --- Normalize lbl to +/- 2 SD and scaled between [0, 1]
    mu = arrays['xs']['dat'].mean() 
    sd = arrays['xs']['dat'].std() 

    lbl = (dat - mu) / (4 * sd) 
    lbl = (lbl + 0.5).clip(min=0, max=1)

    # --- Return dat and lbl arrays 
    return {
        'xs': {'dat': dat}, 
        'ys': {'lbl': lbl}}

# =====================================================================================
# DEFINE VAE ENCODER
# =====================================================================================

def create_encoder(p):
    """
    Method to create simple VAE encoder

    """
    inputs = Input(shape=(1, None, None, 1), dtype='float32')

    # --- Define kwargs dictionary
    kwargs = {
        'padding': 'same',
        'kernel_initializer': 'he_normal'}

    # --- Define lambda functions
    conv = lambda x, filters, strides : layers.Conv3D(filters=filters, strides=strides, kernel_size=(1, 3, 3), **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.ReLU()(x)

    # --- Define stride-1, stride-2 blocks
    conv1 = lambda filters, x : relu(norm(conv(x, filters, strides=1)))
    conv2 = lambda filters, x : relu(norm(conv(x, filters, strides=(1, 2, 2))))

    # --- Define contracting layers
    l1 = conv1(16, inputs)
    l2 = conv1(32, conv2(32, l1))
    l3 = conv1(48, conv2(48, l2))
    l4 = conv1(64, conv2(64, l3))
#     l5 = conv1(128, conv2(128, l4))

    # --- Convert l4 from feature map (1, 4, 4) to vector (1, 1, 1)
    p1 = layers.AveragePooling3D(pool_size=(1, 4, 4))(l4)
#     p1 = layers.AveragePooling3D(pool_size=(1, 2, 2))(l5)

    # --- Define z_mu and z_sd for VAE sampler 
    z_mu = layers.Conv3D(filters=p['latent_dim'], kernel_size=1, name='z_mu', **kwargs)(p1)
    z_sd = layers.Conv3D(filters=p['latent_dim'], kernel_size=1, name='z_sd', **kwargs)(p1)

    # --- Aggregate outputs
    outputs = {
        'z_mu': z_mu,
        'z_sd': z_sd}

    return Model(inputs=inputs, outputs=outputs, name='encoder')

def create_sampler(p):
    """
    Method to create simple VAE sampler

    """
    inputs = {
        'z_mu': Input(shape=(1, None, None, p['latent_dim']), name='z_mu', dtype='float32'),
        'z_sd': Input(shape=(1, None, None, p['latent_dim']), name='z_sd', dtype='float32')}

    # --- Define sampled feature vectors
    samp = custom.VAESampling()([inputs['z_mu'], inputs['z_sd']])

    return Model(inputs=inputs, outputs=samp, name='sampler')

# =====================================================================================
# DEFINE VAE DECODER 
# =====================================================================================

def create_decoder(p):
    """
    Method to create simple VAE decoder

    """
    inputs = Input(shape=(p['latent_dim'],))

    # --- Define kwargs dictionary
    kwargs = {
        'kernel_size': (1, 5, 5),
        'padding': 'same',
        'kernel_initializer': 'he_normal'}

    # --- Define lambda functions
    tran = lambda x, filters, strides : layers.Conv3DTranspose(filters=filters, strides=strides, **kwargs)(x)
    norm = lambda x : layers.BatchNormalization()(x)
    relu = lambda x : layers.ReLU()(x)

    # --- Define conv transpose block
    tran2 = lambda filters, x : relu(norm(tran(x, filters, strides=(1, 2, 2)))) 

    l0 = layers.Dense(4 * 4 * 64, activation='relu')(inputs)
    l1 = layers.Reshape((1, 4, 4, 64))(l0)
    l2 = tran2(48, l1)
    l3 = tran2(32, l2)
    l4 = tran2(16, l3)
#     l0 = layers.Dense(2 * 2 * 128, activation='relu')(inputs)
#     l1 = layers.Reshape((1, 2, 2, 128))(l0)
#     l2 = tran2(64, l1)
#     l3 = tran2(48, l2)
#     l4 = tran2(32, l3)
#     l5 = tran2(16, l4)

    # --- Reconstructed output
    x = layers.Conv3DTranspose(1, activation='sigmoid', name='lbl', **kwargs)(l4)
#     x = layers.Conv3DTranspose(1, activation='sigmoid', name='lbl', **kwargs)(l5)

    return Model(inputs=inputs, outputs=x, name='decoder')

# =====================================================================================
# DEFINE CUSTOM LOSS 
# =====================================================================================

def vae_loss(z_mu, z_sd):

    def loss_l2(y_true, y_pred):
        """
        Method to implement VAE reconstruction (L2) loss scaled by number of pixels in image (32 **2)

        """
        loss = losses.MeanSquaredError()

        return tf.reduce_mean(loss(y_true, y_pred)) * (32 ** 2) 

    def loss_kl(z_mu, z_sd):
        """
        Method to implement VAE KL-loss

        """
        return -0.5 * tf.reduce_mean(1 + z_sd - tf.square(z_mu) - tf.exp(z_sd)) 

    def loss(y_true, y_pred):
        """
        Method to implement combined L2 reconstruction and KL-loss

        """
        l2 = loss_l2(y_true, y_pred)
        kl = loss_kl(z_mu, z_sd)

        return l2 + kl

    return loss

# =====================================================================================
# ASSEMBLE VAE 
# =====================================================================================

def create_vae(p, client):

    # --- Create VAE model components (subgraphs)
    gpus.autoselect()
    encoder = create_encoder(p)
    sampler = create_sampler(p)
    decoder = create_decoder(p)

    # --- Assemble full VAE model
    latent = encoder(inputs['dat'])
    sample = sampler(latent)
    output = decoder(sample)

    # --- Create output dict with correct name to match generator
    outputs = {'lbl': layers.Lambda(lambda x : x, name='lbl')(output)}

    # --- Create VAE model
    vae = Model(inputs=inputs, outputs=outputs)

    return vae, encoder, sampler, decoder, latent

def load_vae(p, client):
    """
    Method to load VAE

      (1) Create encoder, sampler, decoder and VAE
      (3) Load VAE model (propogates to remaining components)

    """
    vae, encoder, sampler, decoder, latent = create_vae(p, client)

    # --- Load existing
    models = glob.glob('{}/*.hdf5'.format(p['output_dir']))
    if len(models) > 0:
        print('Loading existing model: {}'.format(sorted(models)[-1]))
        vae.load_weights(sorted(models)[-1])

    return vae, encoder, sampler, decoder, latent

# ================================================================
# RUNNING INFERENCE 
# ================================================================

def run_latent_plot(p, client):

    _, _, _, decoder, _ = load_vae(p, client)

    ys = []

    for x in np.linspace(-2.0, +2.0, 25):

        y = decoder.predict([x])
        ys.append(y.squeeze())
        print('Running inference: {:03d}'.format(len(ys)), end='\r')

    show(montage(np.stack(ys)))

def run_inference(p, client):
    """
    Method to test efficacy of unsupervised model

      (1) Run encoder on 512 x 512 image ==> 16 x 16 latent features (if patch size == 32 x 32)
      (2) Clean up latent feature matrix (e.g., remove predictions from outside of breast)
      (3) Collapse latent feature matrix into single value prediction (e.g., mean)
      (4) Correlate latent feature vs. ground-truth ratio of FGT / breast

    """
    _, encoder, _, _, _ = load_vae(p, client)

    # --- Reset Client.preprocess(...)
    Client.preprocess = lambda self, arrays, **kwargs : arrays

    # --- Create client
    configs = {'batch': {'fold': p['fold']}}
    client = Client('{}/data/ymls/client-vae-infer.yml'.format(paths['code']), configs=configs)
    test_train, _ = client.create_generators(test=True) 

    # ===========================================================
    # INFERENCE GENERATOR
    # ===========================================================
    # 
    # arrays['xs']['dat'] ==> input data
    # arrays['xs']['msk'] ==> breast mask
    # arrays['ys']['fgt'] ==> ground-truth ratio of FGT / breast
    #
    # ===========================================================

    for xs, ys in test_train:

        # --- Run encoder ONLY
        latent = encoder.predict(xs['dat'])

        # --- Analyze latent

# ================================================================
# CREATE MODEL AND TRAIN
# ================================================================

# --- Find paths
paths = jtools.get_paths('xr/breast-fgt') 

# --- Prepare hyperparams
p = params.load('./hyper.csv')

# --- Set batch size, fold, and input shape
configs = {
    'batch': {
        'size': p['batch_size'],
        'fold': p['fold']},
#         'sampling': {'MLO-normal': 1.0}},
    'specs': {
        'xs': {'dat': {'shape': {'input': [1, p['shape'], p['shape'], 1]}}},
        'ys': {'lbl': {'shape': {'input': [1, p['shape'], p['shape'], 1]}}}}}

# --- Create client
client = Client('{}/data/ymls/client-vae-train.yml'.format(paths['code']), configs=configs)
inputs = client.get_inputs(Input)

# ================================================================
# run_latent_plot(p, client)
# run_inference(p, client)
# ================================================================

# --- Create VAE
vae, encoder, sampler, decoder, latent = load_vae(p, client)

# -- Compile
vae.compile(
    optimizer=optimizers.Adam(learning_rate=p['LR']), 
    loss={'lbl': vae_loss(z_mu=latent['z_mu'], z_sd=latent['z_sd'])},
    experimental_run_tf_function=False)

# --- Learning rate scheduler
lr_scheduler = callbacks.LearningRateScheduler(lambda epoch, lr : lr * p['LR_decay'])

# --- Model checkpoint callback (save every epoch)
model_saver = callbacks.ModelCheckpoint(filepath='%s/model-{epoch:03d}.hdf5' % p['output_dir'])

# --- Tensorboard
log_dir = '{}/jmodels/logdirs/{}'.format(
    os.path.dirname(p['output_dir']),
    os.path.basename(p['output_dir']))

tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)

# --- Create generators
gen_train, gen_valid = client.create_generators()
client.load_data_in_memory()

iterations = p['iterations']
stepsPerEpoch = 1000
epochCount = int(iterations / stepsPerEpoch)

# --- Train
vae.fit(
    x=gen_train,
    epochs=epochCount,
    steps_per_epoch=stepsPerEpoch,
    callbacks=[lr_scheduler, model_saver, tensorboard_callback])

# ================================================================
# arrs = client.get()
# xs, ys = next(gen_train)
# ================================================================
