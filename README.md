
# PixelGAN

Инструмент разработанный для улучшения качества фото и видео.
В нем используется алгоритм  генеративно-состязательной сети.


# Установка и использование
Для начала требуется установить PyTorch с официального сайта.

Устанавливаем зависимости `pip install -r requirements.txt`

Откройте (или создайте, если его нет) файл `.env` и запишите туда `INPUT_PATH=your_path_to_file`, где `your_path_to_file` это путь до вашего файла. Для удобства вы можете располагать файлы в дирректории `General/input`. Если папки `input` не существует, вы можете ее создать.

Так-же в этом файле создайте еще одну переменную `COLLECTION_OF_MANY_FILES=False`. Если значение установлено `False`, то обрабатываться будет только один файл, указанный в `INPUT_PATH`. Если же установлено значение `True`, то вместо пути до файла в переменной `INPUT_PATH` вам нужно указать путь до папки, в которой лежит множество фото/видео.

ВАЖНО: файлы принимаются только с расширениями (jpg, jpeg, png, mp4, avi, mov)

После всех действий выше вы можете запускать файл `app.py`.
Результаты выполнения вы обнаружите в папке `General/results`.
Если вы обрабатывали видео, в этой папке появятся отдельные кадры видео, а в самом конце файл видео с вашим названием и расширением.


# Уникальность нашего проекта
Уникальность продукта обеспечивается использованием объектно-ориентированного программирования (ООП), которое позволяет структурировать код в виде взаимодействующих объектов, повышая его модульность, гибкость и упрощая разработку, тестирование и сопровождение.

Важным фактором является применение Генеративно-состязательной сети (GAN) в качестве алгоритма. GAN способна эффективно обучаться на больших объемах данных и улучшать результаты с каждой новой итерацией обучения, обеспечивая высокую точность и качество генерации изображений.

Наш продукт обладает мультифункциональностью, предоставляя возможность обработки как отдельных изображений/видео, так и целых папок. 


# Задачи на будущее
1. Подбор весов моделей для более качественной обработки видео
2. Построение более легковесной архитектуры 
3. Добавление генеративной модели для улучшения лиц на видео
4. Ускорение работы алгоримта 
5. Непрерывное совершенствование: Постоянно обновлять и совершенствовать модель искусственного интеллекта, периодически обновляя ее новыми данными и учитывая отзывы пользователей. Быть в курсе последних исследований в области методов улучшения качества видео, чтобы внедрять любые новые подходы или достижения

# Результат работы алгоритма До обработки | После обработки
<div align='center'>
  <img height ="250px"  src="original image.png" />
  <a>  </a>
  <img height ="250px"  src="processed image.png" />
</div>

# Итоги Хакатона
<img src="https://github.com/DIMFLIX-HACKATONS/PixelGAN/blob/55581e863a63e96100c9a8a3c3a4a19c47e0b00f/%D1%81%D0%B5%D1%80%D1%82%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%82.png" alt=""/>
