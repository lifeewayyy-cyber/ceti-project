import os
import math
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Получаем путь к директории текущего скрипта
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()

# === ЭТАП 1: Загрузка и квантование яркости (подавление микрошума) ===
def load_and_quantize(image_name, step=16):
    """
    Загружает изображение, приводит к 200x200 px, Grayscale ('L')
    и квантует значения яркости с шагом step=16.
    """
    path = os.path.join(script_dir, image_name)
    img = Image.open(path).resize((200, 200)).convert("L")
    quantized_pixels = [(p // step) * step for p in img.getdata()]
    return img, quantized_pixels

# Реализация алгоритма RLE (кодер и декодер)
def rle_encode(data):
    """Сжимает массив пикселей в пары (count, value)"""
    if not data:
        return []
    compressed = []
    current_pixel = data[0]
    count = 1
    for pixel in data[1:]:
        if pixel == current_pixel:
            count += 1
        else:
            compressed.append((count, current_pixel))
            current_pixel = pixel
            count = 1
    compressed.append((count, current_pixel))
    return compressed

def rle_decode(compressed_data):
    """Разворачивает RLE-пары обратно в исходный массив пикселей"""
    decompressed = []
    for count, pixel in compressed_data:
        decompressed.extend([pixel] * count)
    return decompressed

# Расчёт информационной энтропии Шеннона
def calculate_entropy(data):
    """Вычисляет энтропию H = - sum(p * log2(p))"""
    total = len(data)
    counts = Counter(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# ЕДИНЫЙ ПАЙПЛАЙ ОБРАБОТКИ КАДРА
def process_pipeline(image_name, step=16):
    # 1. Загрузка и квантование
    img, pixels = load_and_quantize(image_name, step=step)
    
    # 2. Сжатие RLE
    rle_pairs = rle_encode(pixels)
    
    # 3. Декодирование и проверка целостности (Lossless check)
    decoded_pixels = rle_decode(rle_pairs)
    is_correct = (pixels == decoded_pixels)
    
    # 4. Расчёт метрик (RAW volume, RLE volume, K, Entropy)
    v_raw = len(pixels) * 1            # 200x200x1 = 40000 байт
    v_rle = len(rle_pairs) * 2         # Каждая пара (count, value) занимает 2 байта
    k_compress = v_raw / v_rle if v_rle > 0 else 0.0
    entropy = calculate_entropy(pixels)
    
    return {
        "name": image_name,
        "img": img,
        "pixels": pixels,
        "rle_pairs": rle_pairs,
        "is_correct": is_correct,
        "v_raw": v_raw,
        "v_rle": v_rle,
        "k_compress": k_compress,
        "entropy": entropy
    }

# Запуск эксперимента и анализ результатов
image_files = ["simple.jpg", "complex.jpg"]
results = {}

print("--- ТЕСТИРОВАНИЕ АЛГОРИТМА RLE И ЭНТРОПИИ ШЕННОНА ---")
for img_name in image_files:
    full_path = os.path.join(script_dir, img_name)
    if os.path.exists(full_path):
        res = process_pipeline(img_name, step=16)
        results[img_name] = res
        print(f"\n[Изображение: {res['name']}]")
        print(f"  Восстановление без потерь (Lossless): {res['is_correct']}")
        print(f"  Энтропия Шеннона H: {res['entropy']:.2f} бит/px")
        print(f"  Объём RAW (V_raw): {res['v_raw']} байт")
        print(f"  Объём RLE (V_rle): {res['v_rle']} байт")
        print(f"  Коэффициент сжатия K: {res['k_compress']:.2f}x")
    else:
        print(f"\nПредупреждение: Файл '{img_name}' не найден по пути {full_path}")

#  Визуализация результатов (Matplotlib Visualizer)
if len(results) == 2:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # Графика для simple.jpg
    axes[0, 0].imshow(results["simple.jpg"]["img"], cmap="gray")
    axes[0, 0].set_title("Простой кадр (simple.jpg)\n200x200 Grayscale")
    axes[0, 0].axis("off")
    
    axes[0, 1].hist(results["simple.jpg"]["pixels"], bins=16, color="#37ebff", edgecolor="black")
    axes[0, 1].set_title(f"Гистограмма яркостей\nH = {results['simple.jpg']['entropy']:.2f} бит/px")
    axes[0, 1].set_ylabel("Частота")
    
    # Графика для complex.jpg
    axes[1, 0].imshow(results["complex.jpg"]["img"], cmap="gray")
    axes[1, 0].set_title("Сложный кадр (complex.jpg)\n200x200 Grayscale")
    axes[1, 0].axis("off")
    
    axes[1, 1].hist(results["complex.jpg"]["pixels"], bins=16, color="#0541f0", edgecolor="black")
    axes[1, 1].set_title(f"Гистограмма яркостей\nH = {results['complex.jpg']['entropy']:.2f} бит/px")
    axes[1, 1].set_ylabel("Частота")
    
    # Сравнительный график коэффициентов сжатия K
    names = ["simple.jpg", "complex.jpg"]
    k_values = [results["simple.jpg"]["k_compress"], results["complex.jpg"]["k_compress"]]
    
    bars = axes[0, 2].bar(names, k_values, color=["#37ebff", "#0541f0"], edgecolor="black")
    axes[0, 2].set_title("Коэффициент сжатия K (V_raw / V_rle)")
    axes[0, 2].set_ylabel("Кратно (x)")
    for bar in bars:
        yval = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval:.2f}x", ha='center', va='bottom')

    # Сравнительный график объемов памяти V_raw vs V_rle
    v_raws = [results["simple.jpg"]["v_raw"], results["complex.jpg"]["v_raw"]]
    v_rles = [results["simple.jpg"]["v_rle"], results["complex.jpg"]["v_rle"]]
    
    x = range(len(names))
    width = 0.35
    axes[1, 2].bar([p - width/2 for p in x], v_raws, width, label="RAW (байты)", color="gray", edgecolor="black")
    axes[1, 2].bar([p + width/2 for p in x], v_rles, width, label="RLE (байты)", color="#0a1e64", edgecolor="black")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(names)
    axes[1, 2].set_title("Сравнение объема памяти (V_raw vs V_rle)")
    axes[1, 2].set_ylabel("Байты")
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()

