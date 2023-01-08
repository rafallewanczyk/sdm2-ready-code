from typing import List

import numpy as np
from gluoncv.data.transforms import video
from gluoncv.model_zoo import get_model
from mxnet import nd
import mxnet


class I3D:
    MODEL_NAME = 'i3d_inceptionv1_kinetics400'

    def __init__(self, feat_ext: bool):
        self.feat_ext = feat_ext
        self.net = get_model(name=self.MODEL_NAME, nclass=400, pretrained=True, feat_ext=feat_ext, num_segments=1,
                             num_crop=1, ctx=mxnet.gpu())
        self.transform_fn = video.VideoGroupValTransform(size=224, mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])

    def _preprocess(self, clip_input: List[np.array]):
        clip_input = self.transform_fn(clip_input)
        clip_input = np.stack(clip_input, axis=0)
        clip_input = clip_input.reshape((-1,) + (32, 3, 224, 224))
        clip_input = np.transpose(clip_input, (0, 2, 1, 3, 4))
        return clip_input

    def inference(self, vid):
        """
       preprocess image to fit i3d model requirements
       :param vid: 32 elements list of np.array with shape (224,224,3): vid_w, vid_h, rgb
       :return: if fet_ext: vid_features: np.array(1,1024) else klass
       """
        clip_input = self._preprocess(vid)
        input_ten = nd.array(np.array(clip_input).astype('float32', copy=False), ctx=mxnet.gpu())
        pred = self.net(input_ten)
        if self.feat_ext:
            return pred

        classes = self.net.classes
        ind = nd.topk(pred, k=1)[0].astype('int')
        return ('\t[%s], with probability %.3f.' % (classes[ind[0].asscalar()], nd.softmax(pred)[0][ind[0]].asscalar()))
