import os
import pandas as pd
from geopandas import GeoDataFrame
from movingpandas import TrajectoryCollection
from shapely.geometry import Point
from datetime import datetime

from mobiml.datasets import Dataset, MOVER_ID, TRAJ_ID


class TestDataset:
    test_dir = os.path.dirname(os.path.realpath(__file__))

    def setup_method(self):
        df = pd.DataFrame(
            [
                {
                    "geometry": Point(0, 3),
                    "txx": datetime(2018, 1, 1, 12, 0, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 3),
                    "txx": datetime(2018, 1, 1, 12, 6, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 6),
                    "txx": datetime(2018, 1, 1, 12, 10, 0),
                    "tid": 1,
                    "mid": "a",
                },
                {
                    "geometry": Point(6, 9),
                    "txx": datetime(2018, 1, 1, 12, 15, 0),
                    "tid": 1,
                    "mid": "a",
                },
            ]
        ).set_index("txx")
        self.gdf = GeoDataFrame(df, crs=31256)

    def test_dataset_from_gdf(self):
        data = Dataset(self.gdf, name="test", traj_id="tid", mover_id="mid")
        assert isinstance(data, Dataset)
        assert data.name == "test"
        assert data.traj_id == "tid"
        assert data.mover_id == "mid"
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)

    def test_dataset_from_csv(self):
        path = os.path.join(self.test_dir, "data/test.csv")
        data = Dataset(
            path,
            name="test",
            traj_id="tid",
            mover_id="mid",
            timestamp="t",
            crs=31256,
        )
        assert isinstance(data, Dataset)
        assert data.name == "test"
        assert data.traj_id == "tid"
        assert data.mover_id == "mid"
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)

    def test_dataset_from_zipped_csv(self):
        path = os.path.join(self.test_dir, "data/test.zip")
        data = Dataset(
            path,
            name="test",
            traj_id="tid",
            mover_id="mid",
            timestamp="t",
            crs=31256,
        )
        assert isinstance(data, Dataset)
        assert data.name == "test"
        assert data.traj_id == "tid"
        assert data.mover_id == "mid"
        assert TRAJ_ID in data.df.columns
        assert MOVER_ID in data.df.columns
        trajs = data.to_trajs()
        assert isinstance(trajs, TrajectoryCollection)

    def test_get_bounds(self):
        data = Dataset(self.gdf, name="test", traj_id="tid", mover_id="mid")
        assert isinstance(data, Dataset)
        bounds = data.get_bounds()
        min_x, min_y, max_x, max_y = 0, 3, 6, 9
        assert (min_x, min_y, max_x, max_y) == bounds
