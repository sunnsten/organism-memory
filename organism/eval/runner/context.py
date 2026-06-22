import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from organism.config import OrganismConfig
from organism.core import Organism

logger = logging.getLogger(__name__)


class RunContext:
    """
    Execution context for an eval scenario run.

    Creates an isolated database and config for each run.
    The DB is preserved after the run for offline analysis.
    """

    def __init__(
        self,
        base_config: Optional[OrganismConfig] = None,
        db_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        test_id: Optional[str] = None,
        mode: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.base_config = base_config or OrganismConfig()
        self._db_path: Optional[str] = None

        if db_path:
            self._db_path = db_path
        else:
            output_dir_path = Path(output_dir or "runs")
            output_dir_path.mkdir(parents=True, exist_ok=True)

            if test_id and mode and run_id:
                db_name = f"eval_{test_id}_{mode}_{run_id.replace(':', '-').replace('.', '-')}.db"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                db_name = f"eval_{test_id or 'unknown'}_{mode or 'unknown'}_{timestamp}.db"

            self._db_path = str(output_dir_path / db_name)
            logger.info("Creating persistent DB at: %s", self._db_path)

        self.config = self._create_isolated_config()
        self.organism: Optional[Organism] = None

    def _create_isolated_config(self) -> OrganismConfig:
        """Return a deep copy of base_config."""
        import copy
        return copy.deepcopy(self.base_config)

    def create_organism(self) -> Organism:
        """Create Organism v2 using Organism.from_config() so the full pipeline
        (embedder, ConsolidationWorker) is wired up.  The DB is isolated to the
        run-specific path so each eval run starts with a clean store.
        """
        if self.organism is None:
            assert self._db_path is not None
            db_path = Path(self._db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            self.organism = Organism.from_config(
                self.config,
                db_path=str(db_path),
                tenant_id="eval",
            )

        return self.organism

    def get_db_path(self) -> str:
        assert self._db_path is not None
        return self._db_path

    def close(self):
        if self.organism is not None:
            self.organism = None
        if self._db_path:
            logger.info("DB preserved at: %s", self._db_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
