import os
import cv2
import glob
import torch
from typing import Tuple
from loguru import logger
from tqdm import tqdm
from environs import Env


from basicsr.utils import imwrite
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.video_util import VideoReader, VideoWriter
from basicsr.archs.rrdbnet_arch import RRDBNet


class GAN:
	def __init__(self, input_path: str, result_root: str = None):

		##==> Input Data
		###############################################
		self.input_path: str = input_path
		self.result_root: str = result_root

		##==> Other Variables
		self._device = get_device()
		self._upscale = 2

		if not gpu_is_available():
			logger.warning("Running on CPU now! Make sure your PyTorch version matches your CUDA.")

	def bg_upsampler(self):
		use_half = False
		if torch.cuda.is_available():
			no_half_gpu_list = ['1650', '1660']
			if True not in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
				use_half = True

		model = RRDBNet(
			num_in_ch=3,
			num_out_ch=3,
			num_feat=64,
			num_block=23,
			num_grow_ch=32,
			scale=2,
		)

		upsampler = RealESRGANer(
			scale=2,
			model_path="General/models/BusinessLunchGAN.pth",
			model=model,
			tile=400,
			tile_pad=40,
			pre_pad=0,
			half=use_half
		)

		return upsampler

	def find_files(self) -> Tuple[bool, dict, list]:
		img_or_video = False  # False - IMG | True - VIDEO
		video: dict = {"name": None, "audio": None, "fps": None}
		img_list: list = []

		if self.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):  # ЕСЛИ ИЗОБРАЖЕНИЕ
			img_or_video = False
			img_list.append(self.input_path)
			self.result_root = f"General/results"

		elif self.input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):  # ЕСЛИ ВИДЕО
			vidr = VideoReader(self.input_path)

			frame = vidr.get_frame()
			while frame is not None:
				img_list.append(frame)
				frame = vidr.get_frame()

			video_name = os.path.basename(self.input_path)[:-4]
			img_or_video = True
			video.update(
				audio=vidr.get_audio(),
				fps=vidr.get_fps(),
				name=video_name
			)
			self.result_root = f"General/results/{video_name}"
			vidr.close()
		else:
			raise FileNotFoundError(
				"Я не понял, какой файл загружать...\n"
				"Мне нужна ссылка на фото (jpg, jpeg, png, JPG, JPEG, PNG),"
				" или на видео (mp4, mov, avi, MP4, MOV, AVI)"
			)

		if len(img_list) == 0:
			raise FileNotFoundError(
				"Входное изображение/видео не найдено...\n"
				"Мне нужна ссылка на фото (jpg, jpeg, png, JPG, JPEG, PNG),"
				" или на видео (mp4, mov, avi, MP4, MOV, AVI)"
			)

		return img_or_video, video, img_list

	def run(self):
		img_or_video, video, img_list = self.find_files()

		for i, img_path in enumerate(img_list, start=1):
			if not img_or_video:  # ВЫВОДИМ ИНФУ О ОБРАБАТЫВАЮЩЕМСЯ ФОТО
				img_name = os.path.basename(img_path)
				basename, ext = os.path.splitext(img_name)
				img = cv2.imread(img_path, cv2.IMREAD_COLOR)
			else:  # ВЫВОДИМ ИНФУ О ОБРАБАТЫВАЮЩЕМСЯ ВИДЕО
				basename = str(i).zfill(6)
				img_name = f"{video['name']}_{basename}"
				img = img_path

			logger.info(f"[{i}/{len(img_list)}] Обработка файла: {img_name}")
			bg_img = self.bg_upsampler().enhance(img, outscale=self._upscale)[0]
			bg_img = cv2.resize(bg_img, (720, 480), interpolation=cv2.INTER_AREA)

			if bg_img is not None:
				if img_or_video:
					save_restore_path = os.path.join(self.result_root, video['name'], f'{basename}.png')
				else:
					save_restore_path = os.path.join(self.result_root, f'{basename}.png')

				imwrite(bg_img, save_restore_path)

		if img_or_video:
			video_frames = []
			img_list = sorted(glob.glob(os.path.join(self.result_root, video['name'], '*.[jp][pn]g')))

			for img_path in tqdm(img_list, desc="Сбор кадров видео"):
				img = cv2.imread(img_path)
				video_frames.append(img)

			height, width = video_frames[0].shape[:2]
			save_restore_path = os.path.join(self.result_root, f"{video['name']}.mp4")
			vidwriter = VideoWriter(save_restore_path, height, width, video['fps'], video['audio'])

			for f in tqdm(video_frames, desc="Склеивание кадров в одно целое"):
				vidwriter.write_frame(f)

			vidwriter.close()


def multi_files(path_to_files):
	for name_file in os.listdir(path_to_files):
		process = GAN(input_path=f'{path_to_files}/{name_file}')
		process.run()


def one_file(path_to_file):
	process = GAN(input_path=path_to_file)
	process.run()


if __name__ == "__main__":
	env = Env()
	env.read_env()

	collection_of_many_files = env.bool("COLLECTION_OF_MANY_FILES")
	path = env.str("INPUT_PATH")

	if collection_of_many_files:
		multi_files(path)
	else:
		one_file(path)
