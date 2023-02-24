import paddle


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).astype('float32') 
    mask /= mask.mean()
    loss = paddle.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()