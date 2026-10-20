
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


class RegressaoLinearSimples:
    def __init__(self):
        self.intercepto = None
        self.coeficiente = None
        self.ajustado = False

    def calcular_coeficientes(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float]:
        # Médias de x e y
        x_media = np.mean(x)
        y_media = np.mean(y)

        # Calcula o coeficiente angular (inclinação)
        numerador = np.sum((x - x_media) * (y - y_media))
        denominador = np.sum((x - x_media) ** 2)

        if denominador == 0:
            raise ValueError("Os valores de x precisam variar.")

        coeficiente = numerador / denominador

        # Calcula o intercepto
        intercepto = y_media - coeficiente * x_media

        return intercepto, coeficiente

    def ajustar(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.size != y.size:
            raise ValueError("x e y precisam ter o mesmo número de valores.")

        if x.size < 2:
            raise ValueError("Informe pelo menos dois pares de dados.")

        self.intercepto, self.coeficiente = (
            self.calcular_coeficientes(x, y)
        )
        self.ajustado = True

        print("✅ Modelo ajustado!")
        print(
            f"Equação: y = {self.intercepto:.2f} "
            f"+ {self.coeficiente:.2f}x"
        )

    def predizer(self, x: np.ndarray) -> np.ndarray:
        if not self.ajustado:
            raise ValueError("Ajuste o modelo antes de fazer predições.")

        return self.intercepto + self.coeficiente * np.asarray(x)

    def calcular_metricas(
        self, x: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        y = np.asarray(y, dtype=float)
        y_pred = self.predizer(x)

        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2,
        }

    def plotar_resultados(
        self, x: np.ndarray, y: np.ndarray,
        titulo: str = "Regressão Linear Simples"
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        y_pred = self.predizer(x)
        residuos = y - y_pred

        # Ordena x para desenhar a linha corretamente
        ordem = np.argsort(x)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Dados e linha de regressão
        axes[0].scatter(x, y, label="Dados reais")
        axes[0].plot(
            x[ordem], y_pred[ordem],
            label="Linha de regressão"
        )
        axes[0].set_title(titulo)
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].legend()
        axes[0].grid(True)

        # 2. Resíduos
        axes[1].scatter(x, residuos)
        axes[1].axhline(y=0, linestyle="--")
        axes[1].set_title("Análise dos resíduos")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Resíduo (real - predito)")
        axes[1].grid(True)

        # 3. Valores reais vs. preditos
        axes[2].scatter(y, y_pred)
        minimo = min(np.min(y), np.min(y_pred))
        maximo = max(np.max(y), np.max(y_pred))
        axes[2].plot([minimo, maximo], [minimo, maximo], "--")
        axes[2].set_title("Reais vs. preditos")
        axes[2].set_xlabel("Valor real")
        axes[2].set_ylabel("Valor predito")
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()


def exemplo_basico():
    # Exemplo do material: área da casa (m²) versus preço (milhares de R$)
    area = np.array([50, 60, 70, 80, 90, 100, 110, 120])
    preco = np.array([150, 180, 210, 240, 270, 300, 330, 360])

    modelo = RegressaoLinearSimples()
    modelo.ajustar(area, preco)

    print(f"Intercepto: R$ {modelo.intercepto:.2f} mil")
    print(
        "A cada m² adicional, o preço estimado aumenta "
        f"R$ {modelo.coeficiente * 1000:.0f}"
    )

    casa_nova = 95
    preco_estimado = modelo.predizer(np.array([casa_nova]))[0]
    print(f"Preço estimado para uma casa de {casa_nova} m²: "
          f"R$ {preco_estimado:.0f} mil")

    metricas = modelo.calcular_metricas(area, preco)
    print("\nMétricas:")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")

    modelo.plotar_resultados(area, preco, "Preço da casa vs. área")


if __name__ == "__main__":
    exemplo_basico()
