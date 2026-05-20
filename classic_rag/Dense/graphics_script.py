import matplotlib.pyplot as plt
import numpy as np

# Данные для реранкеров
models = ['bge-reranker-v2-m3', 'Qwen3-Reranker-0.6B']
precision = [0.4458, 0.3833]
recall = [0.8333, 0.7708]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

# Отрисовка столбцов (используем новые цвета для дифференциации от эмбеддингов)
rects1 = ax.bar(x - width/2, precision, width, label='Precision@5', color='#4EA8DE')
rects2 = ax.bar(x + width/2, recall, width, label='Recall@5', color='#F4A261')

# Настройка осей и оформления
ax.set_ylabel('Значение метрики')
ax.set_title('Сравнение моделей Reranker')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend()

# Добавление подписей с точными значениями над барами
ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

fig.tight_layout()
plt.show()