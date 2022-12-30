import mindspore
from mindspore.ops import constexpr
import numpy as np
from mindspore import ops, context
from mindspore import Tensor
import mindspore
from mindspore import dtype as mstype
import mindspore.numpy as msnp
from mindspore.common.initializer import One, Normal
from mindspore import nn
from mindspore import dtype as mstype

import mindspore
import mindspore.numpy as msnp
import numpy as np
from mindspore import Tensor, context, ms_function
from mindspore import nn
from mindspore import ops
from mindspore.ops import constexpr
from mindspore.ops import functional as F
from mindspore import dtype as mstype
from mindspore.common.initializer import One, Normal
from mindspore.nn.loss.loss import LossBase

context.set_context(mode=context.GRAPH_MODE)


class Softmaxloss(LossBase):
    """Softmaxloss"""

    def __init__(self, sparse=True, smooth_factor=0.1, num_classes=5184):
        super(Softmaxloss, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction="mean")

    def construct(self, logit, label=None):
        """Tripletloss"""
        if not self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss


class Tripletloss(LossBase):
    """Tripletloss"""

    def __init__(self, margin=0.1):
        super(Tripletloss, self).__init__()
        self.margin = margin
        self.sqrt = ops.Sqrt()
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.square = ops.Square()
        self.div = ops.Div()
        self.reshape = ops.Reshape()
        self.split = ops.Split(1, 3)
        self.relu = nn.ReLU()
        self.expand_dims = ops.ExpandDims()
        self.reduce_mean = ops.ReduceMean(keep_dims=False)

    def construct(self, logit, label=None):
        """Tripletloss c"""
        fea_dim = logit.shape[1]
        input_norm = self.sqrt(self.reduce_sum(self.square(logit), 1))
        logit = self.div(logit, input_norm)
        output = self.reshape(logit, (-1, 3, fea_dim))
        anchor, positive, negative = self.split(output)
        anchor = F.reshape(anchor, (-1, fea_dim))
        positive = self.reshape(positive, (-1, fea_dim))
        negative = self.reshape(negative, (-1, fea_dim))
        a_p = self.square(anchor - positive)
        a_n = self.square(anchor - negative)
        a_p = self.reduce_sum(a_p, 1)
        a_n = self.reduce_sum(a_n, 1)
        loss = a_p - a_n + self.margin
        loss = self.relu(loss)
        loss = self.reduce_mean(loss)
        return loss


class mix_loss(nn.Cell):
    def __init__(self):
        super(mix_loss, self).__init__()

        self.max_loss = Softmaxloss()
        self.tri_loss = Tripletloss()

    def construct(self, logits, label):
        feature, classes = logits
        L1 = self.max_loss(classes, label)
        L2 = self.tri_loss(feature, label)
        return L1  + L2 *0.0