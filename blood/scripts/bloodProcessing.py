"""
ЛР5. Визуализация пульсовой волны — обработка (п5.4 и п5.5).

    python3 bloodProcessing.py

Что делает:
  п5.4  калибровка: отсчёты АЦП -> мм рт. ст., график pressure-calibration.png
  п5.5  для каждой записи (в покое и после нагрузки):
        <имя>-pressure.png   давление от времени, отмечены систола и диастола
        <имя>-pulse.png      пульсации на участке 10–30 с с пиками
        <имя>-waves.png      пульсации на всём спуске, волны пронумерованы
        <имя>-amplitude.png  амплитуда волн, систола и диастола
        и печатает пульс и давления (+1 балл: скрипт находит их сам).

Пути к файлам и, при желании, моменты, найденные глазами, — в блоке НАСТРОЙКИ внизу.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ---------------------------------------------------------------- чтение данных
def read(path):
    """Файл Blood Lab -> (отсчёты АЦП, длительность в с, число отсчётов)."""
    duration, samples = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                if "Duration" in line:
                    duration = float(line.split(":")[1].split()[0])
                continue
            samples.append(int(line))
    samples = np.array(samples, dtype=float)
    return samples, duration, len(samples)


def smooth(x, n):
    """Скользящее среднее по n точкам (центрированное).
    На краях делим на число реально попавших в окно точек, чтобы не было
    ложного провала, который даёт np.convolve(..., mode='same') с нулями."""
    k = np.ones(n)
    return np.convolve(x, k, mode="same") / np.convolve(np.ones_like(x), k, mode="same")


# ---------------------------------------------------------------- п5.4 калибровка
def calibration(files, plotPath):
    """files: {давление мм рт. ст.: путь к файлу}. Возвращает fit = [k, b]."""
    pressure = np.array(sorted(files))
    adc = np.array([read(files[P])[0].mean() for P in pressure])
    fit, cov = np.polyfit(adc, pressure, 1, cov=True)
    k, b = fit
    dk, db = np.sqrt(np.diag(cov))
    resid = pressure - np.polyval(fit, adc)

    print(f"Калибровка: P = ({k:.4f} ± {dk:.4f})·N + ({b:.2f} ± {db:.2f}) мм рт. ст.")
    for P, N, r in zip(pressure, adc, resid):
        print(f"   {P:5.0f} мм рт. ст.   N = {N:7.2f}   отклонение {r:+.2f}")

    plt.figure(figsize=(7, 5))
    plt.plot(adc, pressure, "o", label="калибровочные точки")
    x = np.linspace(adc.min() - 20, adc.max() + 20, 100)
    plt.plot(x, np.polyval(fit, x), "-", label=f"P = {k:.4f}·N {b:+.2f}")
    plt.xlabel("Отсчёты АЦП N")
    plt.ylabel("Давление P, мм рт. ст.")
    plt.title("Калибровка датчика давления")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotPath, dpi=200)
    plt.close()
    return fit


# ---------------------------------------------------------------- п5.5 обработка записи
def processing(name, path, fit, plotDir, tSys=None, tDia=None, tValve=None):
    """Обработка одной записи. tSys, tDia, tValve — моменты, найденные глазами;
    если не заданы, скрипт находит их сам."""
    title = {"rest": "в покое", "fitness": "после нагрузки"}.get(name, name)
    # --- 1. Давление
    samples, duration, count = read(path)
    p = np.polyval(fit, samples)
    time = np.linspace(0, duration, count)
    rate = count / duration
    p = smooth(p, int(rate * 0.05))                      # шум квантования, 50 мс

    # --- 2. Пульсации
    win = int(rate * 1.5)
    edge = win // 2
    trend = smooth(p, win)
    osc = p - trend
    peaks, _ = find_peaks(osc, distance=0.4 * rate, prominence=0.3)

    # --- 3. Пульс на участке 10–30 с
    inside = peaks[(time[peaks] > 10) & (time[peaks] < 30)]
    pulse = (len(inside) - 1) * 60 / (time[inside[-1]] - time[inside[0]])
    # проверка: если между соседними пиками есть интервал в 1,5 раза больше
    # типичного, значит удар пропущен (волна слишком мелкая или помеха)
    dt = np.diff(time[inside])
    missed = int(np.sum(np.round(dt / np.median(dt)) - 1))
    pulseLabel = f"{len(inside)} пиков, пульс {pulse:.0f} уд/мин"
    if missed > 0:
        # пульс по медианному интервалу между ударами: пропуски и помехи
        # на него почти не влияют
        print(f"   ВНИМАНИЕ ({name}): на 10–30 с пропущено ~{missed} ударов, по формуле "
              f"(N−1)·60/(tN−t1) пульс занижен до {pulse:.0f}; берём медианный интервал")
        pulse = 60 / np.median(dt)
        pulseLabel = f"пульс {pulse:.0f} уд/мин (по медианному интервалу)"

    # --- 4. Участок спуска: от открытия клапана до 40 мм рт. ст.
    if tValve is None:
        # Клапан открыли там, где давление после максимума начинает спускаться
        # со скоростью хотя бы половину типичной скорости спуска
        # (до этого манжета только медленно оседает после накачки).
        iMax = int(np.argmax(p))
        iEnd = iMax + int(np.argmax(p[iMax:] < 40)) if np.any(p[iMax:] < 40) else count - 1
        slope = -np.gradient(trend, 1 / rate)            # скорость спуска, мм рт. ст./с
        v = np.median(slope[iMax:iEnd])
        iValve = iMax + int(np.argmax(slope[iMax:iEnd] > 0.5 * v))
        tValve = time[iValve]
    # волны ищем с полуокна тренда (0,75 с) после открытия клапана:
    # на изломе тренд запаздывает, и самая первая волна получается искажённой
    i0 = int(tValve * rate) + edge
    i1 = i0 + int(np.argmax(p[i0:] < 40)) if np.any(p[i0:] < 40) else count - edge
    i1 = min(i1, count - edge)

    troughs, _ = find_peaks(-osc, distance=0.4 * rate, prominence=0.3)
    troughs = troughs[(troughs >= i0) & (troughs <= i1)]
    amp, tAmp, top_i = [], [], []
    for a, c in zip(troughs[:-1], troughs[1:]):
        top = a + np.argmax(osc[a:c])                    # вершина между впадинами
        amp.append(osc[top] - osc[a])
        tAmp.append(time[top])
        top_i.append(top)
    amp, tAmp, top_i = np.array(amp), np.array(tAmp), np.array(top_i)

    # выбросы от движения: в 1,5 раза выше медианы соседних волн
    n = len(amp)
    ok = np.ones(n, bool)
    for j in range(n):
        nb = np.r_[max(0, j - 3):j, j + 1:min(n, j + 4)]
        if len(nb) and amp[j] > 1.5 * np.median(amp[nb]):
            ok[j] = False
    # сглаженная амплитуда: медиана по 7 соседним (без выбросов)
    good = np.nonzero(ok)[0]
    ampS = np.full(n, np.nan)
    for gi, j in enumerate(good):
        ampS[j] = np.median(amp[good[max(0, gi - 3):gi + 4]])

    jMax = int(np.nanargmax(ampS))
    level = 0.8 * ampS[jMax]

    # --- автоматический поиск давлений
    # появление волн. Пока артерия пережата, в манжете видны только мелкие
    # одинаковые толчки в её край. Начальный уровень — медиана первых 7 волн.
    # Если он уже больше половины наибольшей амплитуды — волны крупные с самого
    # начала спуска, и появлением считаем первую волну. Иначе появление — первая
    # волна, где сглаженная амплитуда выросла над начальным уровнем на 15 % размаха.
    base = np.median(amp[good[:7]])
    if base >= 0.5 * ampS[jMax]:
        jFirst = int(good[0])
    else:
        thr = base + 0.15 * (ampS[jMax] - base)
        jFirst = int(next(j for j in good if ampS[j] >= thr))
    jSys = min(jFirst + 4, n - 1)                        # 5-я волна (выбросы тоже считаются)
    below = [j for j in good if j > jMax and ampS[j] < level]
    jDia = below[0] if below else good[-1]
    autoSys, autoDia = tAmp[jSys], tAmp[jDia]

    tS = autoSys if tSys is None else tSys
    tD = autoDia if tDia is None else tDia
    pSys = p[int(round(tS * rate))]
    pDia = p[int(round(tD * rate))]

    # --- графики
    # (а) давление с систолой и диастолой
    plt.figure(figsize=(9, 5))
    plt.plot(time, p)
    plt.plot([tS, tD], [pSys, pDia], "r*", ms=12,
             label="давление %.0f/%.0f мм рт. ст." % (pSys, pDia))
    plt.text(tS + 1, pSys + 4, "Систола")
    plt.text(tD + 1, pDia + 4, "Диастола")
    plt.xlabel("Время, с"); plt.ylabel("Давление, мм рт. ст.")
    plt.title(f"Давление в манжете ({title})")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(plotDir, f"{name}-pressure.png"), dpi=200); plt.close()

    # (б) пульсации на 10–30 с
    plt.figure(figsize=(9, 4.5))
    m = (time >= 10) & (time <= 30)
    plt.plot(time[m], osc[m])
    plt.plot(time[inside], osc[inside], "r.", ms=9,
             label=pulseLabel)
    plt.xlim(10, 30)
    plt.xlabel("Время, с"); plt.ylabel("Пульсации, мм рт. ст.")
    plt.title(f"Пульсации ({title})")
    plt.grid(alpha=0.3); plt.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(plotDir, f"{name}-pulse.png"), dpi=200); plt.close()

    # (в) пульсации на всём спуске с номерами волн — чтобы отсчитать 5-ю глазами
    t0, t1 = time[i0], time[i1]
    rows = max(1, int(np.ceil((t1 - t0) / 30)))
    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.2 * rows), squeeze=False)
    for r, ax in enumerate(axes[:, 0]):
        a, bnd = t0 + r * (t1 - t0) / rows, t0 + (r + 1) * (t1 - t0) / rows
        mm = (time >= a) & (time <= bnd)
        ax.plot(time[mm], osc[mm], lw=0.9)
        for num, (ti, ii) in enumerate(zip(tAmp, top_i), start=1):
            if a <= ti <= bnd:
                ax.plot(ti, osc[ii], "r.", ms=7)
                ax.annotate(str(num), (ti, osc[ii]), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=7, color="C3")
        ax.axvline(tS, color="C3", ls="--", lw=1)
        ax.axvline(tD, color="C2", ls="--", lw=1)
        ax.set_xlim(a, bnd); ax.grid(alpha=0.3)
        ax.set_ylabel("ΔP, мм рт. ст.")
    axes[-1, 0].set_xlabel("Время, с")
    axes[0, 0].set_title(f"Пульсации на участке спуска, волны пронумерованы ({title})")
    fig.tight_layout(); fig.savefig(os.path.join(plotDir, f"{name}-waves.png"), dpi=170); plt.close(fig)

    # (г) амплитуда волн
    plt.figure(figsize=(9, 5.5))
    plt.plot(tAmp[ok], amp[ok], "o", color="C0", alpha=0.45, ms=5, label="амплитуда каждой волны")
    if (~ok).any():
        plt.plot(tAmp[~ok], amp[~ok], "o", mfc="none", mec="gray", ms=7, label="выброс от движения (убран)")
    plt.plot(tAmp[good], ampS[good], "-", color="C0", lw=2.2, label="сглаженная амплитуда")
    five = np.arange(jFirst, min(jFirst + 5, n))
    plt.plot(tAmp[five], amp[five], "o", mfc="none", mec="C3", mew=2, ms=9, label="первые пять волн")
    ytop = amp.max() * 1.15
    plt.axvline(tS, color="C3", ls="--", lw=1)
    plt.axvline(tD, color="C2", ls="--", lw=1)
    plt.axhline(level, color="C2", ls="--", lw=1)
    plt.text(tS + 0.5, ytop * 0.97, f"5-я волна:\nсистолическое {pSys:.0f}", color="C3", va="top")
    plt.text(tD + 0.5, ytop * 0.97, f"амплитуда заметно\nуменьшилась:\nдиастолическое {pDia:.0f}",
             color="C2", va="top")
    plt.text(tAmp[-1], level * 1.02, "0,8 от наибольшей", color="C2", ha="right", va="bottom")
    plt.ylim(0, ytop)
    plt.xlabel("Время [с]"); plt.ylabel("Амплитуда волны [мм рт. ст.]")
    plt.title(f"Амплитуда пульсовых волн ({title})")
    plt.grid(alpha=0.3); plt.legend(loc="best")
    plt.tight_layout(); plt.savefig(os.path.join(plotDir, f"{name}-amplitude.png"), dpi=200); plt.close()

    print(f"{name}: пульс {pulse:.0f} уд/мин, давление {pSys:.0f}/{pDia:.0f} мм рт. ст.  "
          f"(систола {tS:.1f} с, диастола {tD:.1f} с; клапан {tValve:.1f} с, волны {t0:.1f}–{t1:.1f} с, "
          f"{(p[i0] - p[i1]) / (t1 - t0):.1f} мм рт. ст./с)")
    if tSys is not None or tDia is not None:
        print(f"   скрипт сам нашёл бы: систола {autoSys:.1f} с ({p[int(autoSys*rate)]:.0f}), "
              f"диастола {autoDia:.1f} с ({p[int(autoDia*rate)]:.0f})")
    return pulse, pSys, pDia


# ================================================================ НАСТРОЙКИ
if __name__ == "__main__":
    import sys
    dataPath = sys.argv[1] if len(sys.argv) > 1 else "data/"
    plotDir = sys.argv[2] if len(sys.argv) > 2 else "plots/"
    os.makedirs(plotDir, exist_ok=True)

    # Калибровка: {давление, мм рт. ст.: имя файла}
    calFiles = {P: f"calibr_{P}.txt" for P in (40, 60, 80, 100, 120, 140)}
    # Записи: имя для графиков -> (файл, tSys, tDia, tValve).
    # None — скрипт находит момент сам; число — момент, найденный глазами, с.
    # tValve — начало ровного спуска: на наших записях до него клапан
    # крутили ступеньками, и эти скачки нельзя принимать за пульсовые волны.
    records = {
        "rest":    ("vova_90_c_160.txt", None, None, 22.0),
        "fitness": ("vova_90_f_160.txt", None, None, 12.8),
    }
    # Для примера данных из стартового набора:
    # calFiles = {P: f"{P} mmHg.txt" for P in (40, 80, 120, 160)}
    # records = {"rest": ("rest.txt", None, None, None),
    #            "fitness": ("fitness.txt", None, None, None)}

    fit = calibration({P: os.path.join(dataPath, f) for P, f in calFiles.items()},
                      os.path.join(plotDir, "pressure-calibration.png"))
    results = {}
    for name, (f, tS, tD, tV) in records.items():
        results[name] = processing(name, os.path.join(dataPath, f), fit, plotDir,
                                   tSys=tS, tDia=tD, tValve=tV)

    labels = {"rest": "В покое", "fitness": "После нагрузки"}
    print("\n| Запись | Пульс, уд/мин | Давление, мм рт. ст. |")
    print("| --- | :---: | :---: |")
    for name, (pulse, ps, pd) in results.items():
        print(f"| {labels.get(name, name)} | {pulse:.0f} | {ps:.0f}/{pd:.0f} |")
