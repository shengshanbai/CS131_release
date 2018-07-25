import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for Ho in range(Hi):
        for Wo in range(Wi):
            for pixH in range(Hk):
                for pixW in range(Wk):
                    imageH=Hk//2-pixH+Ho
                    imageW=Wk//2-pixW+Wo
                    imagePix=0
                    if imageH >=0 and imageH<Hi and imageW>=0 and imageW<Wi:
                        imagePix=image[imageH,imageW]
                    out[Ho,Wo]+=kernel[pixH,pixW]*imagePix
                    
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = np.zeros((H+pad_height*2,W+pad_width*2))

    ### YOUR CODE HERE
    out[pad_height:H+pad_height,pad_width:W+pad_width]=image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    weightM=np.flipud(np.fliplr(kernel))
    imagePad=zero_pad(image,Hk//2,Wk//2)
    for Ho in range(Hi):
        for Wo in range(Wi):
            out[Ho,Wo]=np.sum(imagePad[Ho:Ho+Hk,Wo:Wo+Wk]*weightM)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image = zero_pad(image, Hk//2, Wk//2)
    kernel = np.flip(np.flip(kernel, 0), 1)
    # The trick is to lay out all the (Hk, Wk) patches and organize them into a (Hi*Wi, Hk*Wk) matrix.
    # Also consider the kernel as (Hk*Wk, 1) vector. Then the convolution naturally reduces to a matrix multiplication.
    mat = np.zeros((Hi*Wi, Hk*Wk))
    for i in range(Hi*Wi):
        row = i // Wi
        col = i % Wi
        mat[i, :] = image[row: row+Hk, col: col+Wk].reshape(1, Hk*Wk)
    out = mat.dot(kernel.reshape(Hk*Wk, 1)).reshape(Hi, Wi)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    kernel=np.flip(np.flip(g, 0), 1)
    out=conv_fast(f,kernel)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g=g-np.mean(g)
    out=cross_correlation(f,g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    
    out = None
    ### YOUR CODE HERE
    image=f
    kernel=(g-np.mean(g))/np.std(g)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    imagePad=zero_pad(image,Hk//2,Wk//2)
    for Ho in range(Hi):
        for Wo in range(Wi):
            imagePatch=imagePad[Ho:Ho+Hk,Wo:Wo+Wk]
            imagePatch=(imagePatch-np.mean(imagePatch))/np.std(imagePatch)
            out[Ho,Wo]=np.sum(imagePatch*kernel)
    ### END YOUR CODE

    return out
