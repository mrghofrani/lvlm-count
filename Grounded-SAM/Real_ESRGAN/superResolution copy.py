import os
import sys
import glob
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
# Assuming 'MyCodes' is your base directory
my_codes_path = os.path.abspath('../../Real_ESRGAN')  
# print(my_codes_path)
sys.path.append(my_codes_path)
# Get the absolute path of the current working directory
current_directory = os.getcwd() 
# print(f"Current working: ", current_directory)
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))
# print("this is root dir :",ROOT_DIR)
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# print("this is root dir :",ROOT_DIR)
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class SuperResolution:
    def __init__(self, model_name='RealESRGAN_x4plus', denoise_strength=0.5, outscale=4, model_path=None, tile=0,
                 tile_pad=10, pre_pad=0, face_enhance=False, fp32=False, alpha_upsampler='realesrgan', ext='auto',
                 gpu_id=None):
        self.model_name = model_name.split('.')[0]
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.model_path = model_path
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.face_enhance = face_enhance
        self.fp32 = fp32
        self.alpha_upsampler = alpha_upsampler
        self.ext = ext
        self.gpu_id = gpu_id

        # Determine models according to model names
        if self.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self.netscale = 4
            self.file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

        # Determine model paths
        if self.model_path is not None:
            self.model_path = self.model_path
        else:
            self.model_path = os.path.join('weights', self.model_name + '.pth')
            if not os.path.isfile(self.model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in self.file_url:
                    # model_path will be updated
                    self.model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

        # Use dni to control the denoise strength
        self.dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = self.model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            self.model_path = [self.model_path, wdn_model_path]
            self.dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        # Restorer
        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            dni_weight=self.dni_weight,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id)

        if self.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            self.face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self.upsampler)

    def superize(self, img_path, output_dir='results', suffix='out'):
        os.makedirs(output_dir, exist_ok=True)

        if os.path.isfile(img_path):
            paths = [img_path]
        else:
            paths = sorted(glob.glob(os.path.join(img_path, '*')))

        for idx, img_path in enumerate(paths):
            imgname, extension = os.path.splitext(os.path.basename(img_path))
            print('Processing', idx, imgname)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            elif len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img_mode = None
            else:
                img_mode = None

            try:
                if self.face_enhance:
                    _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if self.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = self.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if suffix == '':
                    save_path = os.path.join(output_dir, f'{imgname}.{extension}')
                else:
                    save_path = os.path.join(output_dir, f'{imgname}_{suffix}.{extension}')
                cv2.imwrite(save_path, output)
