import mxnet as mx 
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from symbol import get_generator, get_discriminator
import os

BATCH_SIZE = 128
#Hyperperameters
SIGMA = 0.02
LR = 0.0002
BETA1 = 0.5
CTX = mx.gpu(0)


#Takes the images in our batch and arranges them in an array so that they can be
#Plotted using matplotlib
def fill_buf(buf, num_images, img, shape):
    width = buf.shape[0]/shape[1]
    height = buf.shape[1]/shape[0]
    img_width = (num_images%width)*shape[0]
    img_hight = (num_images/height)*shape[1]
    buf[img_hight:img_hight+shape[1], img_width:img_width+shape[0], :] = img

#Plots two images side by side using matplotlib
def visualize(root, fake, real):
    #64x3x64x64 to 64x64x64x3
    fake = fake.transpose((0, 2, 3, 1))
    #Pixel values from 0-255
    fake = np.clip((fake+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    #Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    
    #Create buffer array that will hold all the images in our batch
    #Fill the buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n*fake.shape[1]), int(n*fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n*real.shape[1]), int(n*real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])
        
    #Create a matplotlib figure with two subplots: one for the real and the other for the fake
    #fill each plot with our buffer array, which creates the image
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(rbuff)
    name = root+str(time.time())+str(".png")
    fig.savefig(name)   # save the figure to file
    plt.close(fig)
    
# define random iter
class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        #Returns random numbers from a gaussian (normal) distribution 
        #with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]
    
def getData(path="data/AnmineFaceSmall.rec", shape=(3, 64,64), batch_size=8):
    rand_iter = RandIter(BATCH_SIZE, 100)
    image_iter = mx.io.ImageRecordIter(
        path_imgrec=path,  # the target record file
        data_shape=shape,  # output data shape. An 32x128 image
        label_width=1,  # label length, 1 dimension
        shuffle=True,  # shuffle the dataset every epoch or not
        batch_size=batch_size,  # number of samples per batch
        preprocess_threads=4,
        scale = 2./255. 
    )
    return rand_iter, image_iter

def getMods(gen, dis, rand_iter, image_iter, ctx, sigma, lr, beta1):
    #=============Generator Module=============
    generator = mx.mod.Module(symbol=gen, data_names=('rand',), label_names=None, context=ctx)
    generator.bind(data_shapes=rand_iter.provide_data)
    generator.init_params(initializer=mx.init.Normal(sigma))
    generator.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'beta1': beta1,
        })
    mods = [generator]

    # =============Discriminator Module=============
    discriminator = mx.mod.Module(symbol=dis, data_names=('data',), label_names=('label',), context=ctx)
    discriminator.bind(data_shapes=image_iter.provide_data,
              label_shapes=[('label', (BATCH_SIZE,))],
              inputs_need_grad=True)
    discriminator.init_params(initializer=mx.init.Normal(sigma))
    discriminator.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'beta1': beta1,
        })
    mods.append(discriminator)
    return mods

def cal_eval(o, label):
    # loss1
    out_dis = o.asnumpy()
    label = label.asnumpy()
    label[label > 0.5] = 1
    label[label <= 0.5] = 0
    out_dis[out_dis>0.5]=1
    out_dis[out_dis<=0.5]=0
    acc = np.mean((out_dis==label).flatten())
    return acc
    
def train(mods, rand_iter, image_iter, batch_size, ctx, epochs, save_root):
    generator, discriminator = mods
    print('Training...')
    for epoch in range(epochs):
        image_iter.reset()
        for i, batch in enumerate(image_iter):
            #Get a batch of random numbers to generate an image from the generator
            rbatch = rand_iter.next()
            #Forward pass on training batch
            generator.forward(rbatch, is_train=True)
            #Output of training batch is the 64x64x3 image
            outG = generator.get_outputs()

            # Pass the generated (fake) image through the discriminator, and save the gradient
            # Label (for logistic regression) is an array of 0's since this image is fake
            # soft noisy label, and flip the label sometimes
            if i % 500 != 0:
                label = mx.nd.random_uniform(0, 0.3, (batch_size,), ctx)
            else:
                label = mx.nd.random_uniform(0.7, 1.2, (batch_size,), ctx)
            #Forward pass on the output of the discriminator network
            discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            #Do the backwards pass and save the gradient
            discriminator.backward()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

            # acc1
            out_dis1 = discriminator.get_outputs()[0]
            acc1 = cal_eval(out_dis1, label)
            
            #Pass a batch of real images from MNIST through the discriminator
            #Set the label to be an array of 1's because these are the real images
            # soft noisy label
            if i % 500 != 0:
                label[:] = mx.nd.random_uniform(0.7, 1.2, label.shape, ctx)
            else:
                label[:] = mx.nd.random_uniform(0.0, 0.3, label.shape, ctx)
            label[:] = mx.nd.random_uniform(0.7, 1.2, label.shape, ctx)
            data = batch.data[0]-1
            batch.label = [label]
            batch.data = [data]
            #Forward pass on a batch of MNIST images
            discriminator.forward(batch, is_train=True)
            
            # acc2
            out_dis2 = discriminator.get_outputs()[0]
            acc2 = cal_eval(out_dis2, label)

            #Do the backwards pass and add the saved gradient from the fake images to the gradient
            #generated by this backwards pass on the real images
            discriminator.backward()
            for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            #Update gradient on the discriminator 
            discriminator.update()
            
            #Now that we've updated the discriminator, let's update the generator
            #First do a forward pass and backwards pass on the newly updated discriminator
            #With the current batch
            discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
            discriminator.backward()
            #Get the input gradient from the backwards pass on the discriminator,
            #and use it to do the backwards pass on the generator
            diffD = discriminator.get_input_grads()
            generator.backward(diffD)
            #Update the gradients on the generator
            generator.update()

            #Increment to the next batch, printing every 50 batches
            i += 1
            if i % 500 == 0:
                #print('epoch:', epoch, 'iter:', i)
                #print
                #print("   From generator:        From MNIST:")
                print ("Fake input acc: {} (the lower the better the generator), Real input acc: {} (the higher the better the dis is)".format(acc1, acc2))
                visualize(save_root, outG[0].asnumpy(), batch.data[0].asnumpy())
        generator.save_checkpoint(save_root+"Generator", epoch)
        discriminator.save_checkpoint(save_root+"Discriminator", epoch)

def run_train():
    g = get_generator()
    d = get_discriminator()
    tname = time.time()
    root = "save_img"+str(tname)+"/"
    rand_iter, image_iter = getData("data/AnmineFace.rec", (3,64,64), BATCH_SIZE)
    mods = getMods(g, d, rand_iter, image_iter, CTX, SIGMA, LR, BETA1)
    if not os.path.exists(root):
        os.makedirs(root)
    train(mods, rand_iter, image_iter, BATCH_SIZE, CTX, 1000, root)
    
if __name__ == "__main__":
    run_train()