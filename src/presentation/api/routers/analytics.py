import asyncio
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Ścieżkę do analyzera dostosujemy, kiedy zechcesz uporządkować folder analytics
from analytics.analyze_chickens import ChickenDataAnalyzer 

router = APIRouter(tags=["analytics"])


@router.get(
    "/analytics/plot",
    summary="Analytics dashboard as PNG",
)
async def get_analytics_plot(cam: str, detect: str):
    csv_path = f"archive_logs/{cam}_{detect}.csv"
    analyzer = ChickenDataAnalyzer(csv_path)

    data_loaded = await asyncio.to_thread(analyzer.load_data)
    if not data_loaded:
        raise HTTPException(
            status_code=404,
            detail="Brak danych do wygenerowania wykresu.",
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    analyzer.plot_activity_timeline(ax1)
    analyzer.plot_spatial_heatmap(ax2)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)
    plt.close(fig)

    return StreamingResponse(img_buf, media_type="image/png")