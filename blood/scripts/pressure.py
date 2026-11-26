"""
Перевод показаний АЦП в мм рт. ст., сглаживание скользящим средним 50 мс
и построение графика давления от времени.

Запуск:
    python3 pressure.py <папка_с_файлами>
Калибровочные файлы: *calibr_<давление>.txt, файл измерения: *vova_c_160.txt
"""
import re
import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


def read_adc(path):
    """Читает файл Blood Lab: возвращает (массив отсчётов АЦП, длительность в с)."""
    duration = None
    values = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                m = re.search(r"Duration:\s*([\d.]+)", line)
                if m:
                    duration = float(m.group(1))
                continue
            values.append(int(line))
    return np.array(values, dtype=float), duration


def calibrate(folder):
    """Строит линейную калибровку P = k * ADC + b по калибровочным файлам."""
    p_list, adc_list = [], []
    for path in sorted(glob.glob(os.path.join(folder, "*calibr_*.txt"))):
        p = float(re.search(r"calibr_(\d+)", path).group(1))
        adc, _ = read_adc(path)
        p_list.append(p)
        adc_list.append(adc.mean())
    p = np.array(p_list)
    adc = np.array(adc_list)
    order = np.argsort(p)
    p, adc = p[order], adc[order]
    k, b = np.polyfit(adc, p, 1)
    return k, b, adc, p


def adc_to_mmhg(adc, k, b):
    """Перевод показаний АЦП в мм рт. ст."""
    return k * adc + b


def moving_average(x, n):
    """Центрированное скользящее среднее по n точкам (края усредняются по имеющимся точкам)."""
    kernel = np.ones(n)
    s = np.convolve(x, kernel, mode="same")
    cnt = np.convolve(np.ones_like(x), kernel, mode="same")
    return s / cnt


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."

    # 1. Калибровка
    k, b, adc_cal, p_cal = calibrate(folder)
    resid = p_cal - adc_to_mmhg(adc_cal, k, b)
    print(f"Калибровка: P = {k:.5f} * ADC + ({b:.3f})  [мм рт. ст.]")
    print(f"Чувствительность: {1/k:.4f} отсчёта АЦП на 1 мм рт. ст.")
    print("Точки калибровки (P, среднее АЦП, остаток):")
    for pi, ai, ri in zip(p_cal, adc_cal, resid):
        print(f"  {pi:5.0f}  {ai:8.3f}  {ri:+.3f}")

    # 2. Измерение
    meas_path = glob.glob(os.path.join(folder, "*vova_c_160*.txt"))[0]
    adc, duration = read_adc(meas_path)
    fs = len(adc) / duration                      # частота дискретизации, Гц
    t = np.arange(len(adc)) / fs                  # время, с
    p_raw = adc_to_mmhg(adc, k, b)

    # 3. Сглаживание скользящим средним 50 мс
    n = max(1, int(round(0.050 * fs)))
    p_smooth = moving_average(p_raw, n)
    print(f"Частота дискретизации: {fs:.1f} Гц, окно 50 мс = {n} точек")

    # 4. Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [1, 2]})

    ax1.plot(adc_cal, p_cal, "o", label="калибровочные точки")
    xx = np.linspace(adc_cal.min() - 20, adc_cal.max() + 20, 100)
    ax1.plot(xx, adc_to_mmhg(xx, k, b), "-", label=f"P = {k:.4f}·N {b:+.2f}")
    ax1.set_xlabel("Показания АЦП N, отсчёты")
    ax1.set_ylabel("Давление P, мм рт. ст.")
    ax1.set_title("Калибровочная прямая")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(t, p_raw, color="lightgray", lw=0.6, label="исходные данные")
    ax2.plot(t, p_smooth, color="C0", lw=1.4, label="скользящее среднее, 50 мс")
    ax2.set_xlabel("Время t, с")
    ax2.set_ylabel("Давление P, мм рт. ст.")
    ax2.set_title("Давление в манжете от времени")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("pressure_vs_time.png", dpi=200)

    # Сохранение пересчитанных данных
    np.savetxt("pressure_mmHg.csv",
               np.column_stack([t, adc, p_raw, p_smooth]),
               delimiter=",", fmt="%.5f",
               header="t_s,ADC,P_raw_mmHg,P_smooth_mmHg", comments="")
    print("Сохранено: pressure_vs_time.png, pressure_mmHg.csv")
