import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import warnings

# Отключаем специфичные для facenet-pytorch предупреждения
warnings.filterwarnings("ignore", category=UserWarning, module='facenet_pytorch.models.utils.detect_face')


class FaceVerifier:
    """
    Класс для верификации лиц с использованием предобученных моделей MTCNN и FaceNet.
    """

    def __init__(self, device=None):
        """
        Инициализирует детектор лиц (MTCNN) и модель для получения эмбеддингов (FaceNet).
        Модели автоматически скачиваются при первом запуске.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Используется устройство: {self.device}")

        # MTCNN для нахождения лиц на фото
        # keep_all=False вернет только одно, самое уверенно распознанное лицо
        self.mtcnn = MTCNN(
            image_size=160, margin=10, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device, keep_all=False
        )

        # FaceNet (InceptionResnetV1), предобученная на датасете vggface2
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, image_path: str):
        """
        Получает эмбеддинг для лица на изображении.
        :param image_path: Путь к файлу изображения.
        :return: Тензор эмбеддинга или None, если лицо не найдено.
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Ошибка: Файл не найден по пути {image_path}")
            return None

        # Детекция лица, возвращается тензор обрезанного лица
        face_tensor = self.mtcnn(img)

        if face_tensor is None:
            # print(f"Предупреждение: На изображении {image_path} лицо не найдено.")
            return None

        # Добавляем batch dimension (1, 3, 160, 160) и отправляем на устройство
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # Получение эмбеддинга без вычисления градиентов
        with torch.no_grad():
            embedding = self.resnet(face_tensor)

        return embedding.detach()

    @staticmethod
    def calculate_distance(emb1, emb2):
        """
        Вычисляет евклидово расстояние между двумя эмбеддингами.
        :param emb1: Первый эмбеддинг.
        :param emb2: Второй эмбеддинг.
        :return: Расстояние (float) или бесконечность, если один из эмбеддингов None.
        """
        if emb1 is None or emb2 is None:
            return float('inf')
        # .norm() вычисляет L2 норму (евклидово расстояние) разницы векторов
        return (emb1 - emb2).norm().item()