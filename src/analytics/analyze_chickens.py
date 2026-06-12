import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ChickenDataAnalyzer:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.center_xs = []
        self.center_ys = []

    def load_data(self) -> bool:
        """Wczytuje plik CSV i parsuje daty oraz współrzędne."""
        if not os.path.exists(self.csv_path):
            print(f"❌ [Analyzer] Plik {self.csv_path} nie istnieje. Zbierz najpierw dane z detektora!")
            return False
        
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self._extract_bounding_boxes()
            return True
        except Exception as e:
            print(f"💥 [Analyzer] Błąd podczas ładowania danych: {e}")
            return False

    def _extract_bounding_boxes(self):
        """Prywatna metoda parsująca JSONy z ramkami i licząca punkty centralne kurek."""
        self.center_xs.clear()
        self.center_ys.clear()
        
        for _, row in self.df.iterrows():
            try:
                boxes = json.loads(row['bounding_boxes'])
                for box in boxes:
                    # Pobieramy wartości bezpośrednio po kluczach ze słownika
                    x1 = box["x1"]
                    y1 = box["y1"]
                    x2 = box["x2"]
                    y2 = box["y2"]
                    
                    # Wyznaczamy geometryczny środek kury
                    self.center_xs.append((x1 + x2) / 2)
                    self.center_ys.append((y1 + y2) / 2)
            except (json.JSONDecodeError, TypeError, KeyError):
                # Dodałem 'KeyError', żeby skrypt nie padł, jeśli z jakiegoś powodu
                # detektor wypluje słownik bez klucza "x1"
                continue

    def plot_activity_timeline(self, ax):
        """Rysuje wykres aktywności stada na podanym obiekcie Axes."""
        ax.plot(self.df['timestamp'], self.df['count'], color='#ffa500', linewidth=2, label='Kury w kadrze')
        ax.fill_between(self.df['timestamp'], self.df['count'], color='#ffa500', alpha=0.2)
        ax.set_title("Rytm dobowy kurek")
        ax.set_xlabel("Czas")
        ax.set_ylabel("Liczba wykrytych obiektów")
        ax.grid(True, linestyle='--', alpha=0.5)

    def plot_spatial_heatmap(self, ax):
        """Rysuje mapę zagęszczenia (heatmapę) na podanym obiekcie Axes."""
        if not self.center_xs or not self.center_ys:
            ax.text(0.5, 0.5, "Brak danych o pozycjach", ha='center', va='center')
            return

        # Generujemy gęstość rozkładu (KDE)
        sns.kdeplot(x=self.center_xs, y=self.center_ys, cmap="YlOrRd", fill=True, thresh=0.05, ax=ax, alpha=0.8)
        # Nakładamy delikatne punkty, żeby widzieć pojedyncze zdarzenia
        ax.scatter(self.center_xs, self.center_ys, color='black', s=4, alpha=0.2)
        
        ax.set_title("Mapa ulubionych miejsc na wybiegu")
        ax.set_xlabel("Szerokość klatki (px)")
        ax.set_ylabel("Wysokość klatki (px)")
        ax.invert_yaxis()  # Odwracamy Y, bo (0,0) w grafice to lewy górny róg
        ax.grid(True, alpha=0.2)

    def display_dashboard(self, save_to_file: str = None):
        """Generuje i wyświetla (lub zapisuje) pełny dashboard."""
        if self.df is None or self.df.empty:
            print("⚠️ [Analyzer] Brak danych. Uruchom najpierw .load_data()")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        self.plot_activity_timeline(ax1)
        self.plot_spatial_heatmap(ax2)
        
        plt.tight_layout()
        
        if save_to_file:
            plt.savefig(save_to_file)
            print(f"💾 Dashboard zapisany jako obraz: {save_to_file}")
        else:
            plt.show()



    def plot_spatial_heatmap_on_frame(
        self,
        ax,
        background_path: str,
        detection_type: str | None = None,
        scene_id: str | None = None,
    ):
        import cv2
        import seaborn as sns

        df = self.df.copy()

        if detection_type:
            df = df[df["detection_type"] == detection_type]

        if scene_id:
            df = df[df["scene_id"] == scene_id]

        if df.empty:
            ax.text(0.5, 0.5, "Brak danych dla tej sceny", ha="center", va="center")
            return

        bg = cv2.imread(background_path)
        if bg is not None:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            ax.imshow(bg)

        xs = df["cx"].astype(float)
        ys = df["cy"].astype(float)

        sns.kdeplot(
            x=xs,
            y=ys,
            fill=True,
            thresh=0.05,
            alpha=0.45,
            ax=ax
        )

        ax.scatter(xs, ys, s=6, alpha=0.25)

        frame_width = int(df["frame_width"].dropna().iloc[0])
        frame_height = int(df["frame_height"].dropna().iloc[0])

        ax.set_xlim(0, frame_width)
        ax.set_ylim(frame_height, 0)
        ax.set_title(f"Heatmapa: {detection_type or 'wszystko'} | {scene_id or 'bez scene_id'}")
        ax.set_xlabel("X [px]")
        ax.set_ylabel("Y [px]")