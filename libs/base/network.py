from openvino.inference_engine import IENetwork, IECore
import cv2 
import numpy as np 

class nn:
    _N = 0
    _C = 1
    _H = 2
    _W = 3

    def __init__(self, path2model:str, device='CPU') -> None:
        self.ie = IECore()
        self.net  = self.ie.read_network(path2model+'.xml', path2model+'.bin')   # ex: pedestrian-detection-adas-0002
        self.input_name  = next(iter(self.net.input_info))                     # Input blob name "data"
        self.input_shape = self.net.input_info[self.input_name].input_data.shape           # [1,c,h,w]
        self.out_name    = next(iter(self.net.outputs))                    # Output blob name "detection_out"
        self.out_shape   = self.net.outputs[self.out_name].shape            # [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        self.exec_net    = self.ie.load_network(self.net, device)

        self.curr_id = 0
        self.next_id = 1

    def preprocess(self, images: np.ndarray) -> np.ndarray:
        img_tensor = cv2.resize(images, (self.input_shape[nn._W], self.input_shape[nn._H]))
        img_tensor = img_tensor.transpose((2, 0, 1))
        img_tensor = img_tensor.reshape(self.input_shape)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        return img_tensor

    def sync_infer(self, images):
        return self.exec_net.infer(inputs={self.input_name: self.preprocess(images)}) 

    def _async_infer(self, images, request_id):
        self.exec_net.start_async(request_id=request_id, 
                                  inputs={self.input_name: self.preprocess(images)})
        return

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_net.requests[request_id].wait(-1)
        return status
        
    def extract_output(self, request_id):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_net.requests[request_id].output_blobs

    def async_infer(self, images):
        self._async_infer(images, self.curr_id)
        stt = self.wait(request_id= self.curr_id)
        if stt == 0:
            output = self.extract_output(self.curr_id)
            return output 
        else:
            return None 

