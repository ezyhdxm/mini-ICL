"""
Experiment indexing system for searching and browsing experiment results.

This module provides functionality to:
1. Index experiments from results directories into a database
2. Search experiments by configuration parameters
3. Generate a web UI for browsing experiments
"""
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from ml_collections import ConfigDict


class ExperimentIndex:
    """Database for indexing experiment configurations and results."""
    
    def __init__(self, db_path: str = "results/experiment_index.db"):
        """
        Initialize the experiment index database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_name TEXT UNIQUE NOT NULL,
                exp_path TEXT NOT NULL,
                task_name TEXT,
                config_hash TEXT,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Configuration parameters table (key-value pairs)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config_params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exp_id INTEGER NOT NULL,
                category TEXT NOT NULL,  -- 'task', 'model', 'training', etc.
                param_name TEXT NOT NULL,
                param_value TEXT,
                FOREIGN KEY (exp_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(exp_id, category, param_name)
            )
        """)
        
        # Create indices for faster searches
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_name ON experiments(exp_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_config_hash ON experiments(config_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category_param ON config_params(category, param_name, param_value)")
        
        conn.commit()
        conn.close()
    
    def _extract_key_params(self, config: ConfigDict) -> Dict[str, Dict[str, Any]]:
        """
        Extract key parameters from config, organized by category.
        
        Args:
            config: Configuration object
        
        Returns:
            Dictionary with categories as keys and parameter dictionaries as values
        """
        params = {}
        
        # Task parameters
        if hasattr(config, 'task'):
            params['task'] = {
                'name': getattr(config.task, 'name', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'seq_len': getattr(config, 'seq_len', None),
                'alpha': getattr(config.task, 'alpha', None),
                'order': getattr(config.task, 'order', None),
                'n_tasks': getattr(config.task, 'n_tasks', None),
                'n_minor_tasks': getattr(config.task, 'n_minor_tasks', None),
                'n_points': getattr(config.task, 'n_points', None),  # For linear tasks
                'total_trans': getattr(config.task, 'total_trans', None),
                'stationary': getattr(config.task, 'stationary', None),
                'pad': getattr(config.task, 'pad', None),
                'ood': getattr(config.task, 'ood', None),
            }
            # Remove None values
            params['task'] = {k: v for k, v in params['task'].items() if v is not None}
        
        # Model parameters
        if hasattr(config, 'model'):
            params['model'] = {
                'emb_dim': getattr(config.model, 'emb_dim', getattr(config.model, 'n_embd', None)),
                'num_layers': getattr(config.model, 'num_layers', getattr(config.model, 'n_layer', None)),
                'ff_dim': getattr(config.model, 'ff_dim', None),
                'num_heads': str(getattr(config.model, 'num_heads', getattr(config.model, 'n_head', None))),  # Convert to string for tuple/list
                'n_points': getattr(config.model, 'n_points', None),  # For linear tasks
                'dropout': getattr(config.model, 'dropout', None),
                'layer_norm': getattr(config.model, 'layer_norm', None),
                'pos_enc': getattr(config.model, 'pos_enc', None),
            }
            params['model'] = {k: v for k, v in params['model'].items() if v is not None}
        
        # Training parameters
        if hasattr(config, 'training'):
            params['training'] = {
                'num_epochs': getattr(config.training, 'num_epochs', getattr(config.training, 'total_steps', None)),
                'learning_rate': getattr(config.training, 'learning_rate', getattr(config.training, 'lr', None)),
                'warmup_steps': getattr(config.training, 'warmup_steps', None),
                'weight_decay': getattr(config.training, 'weight_decay', None),
                'batch_size': getattr(config.training, 'batch_size', None) or getattr(config, 'batch_size', None),
                'scheduler': getattr(config.training, 'scheduler', None),
            }
            params['training'] = {k: v for k, v in params['training'].items() if v is not None}
        
        # Ensure task dict exists for additional parameters
        if hasattr(config, 'task'):
            if 'task' not in params:
                params['task'] = {}
            # Get batch_size from task if available (for linear tasks, batch_size is in task)
            if hasattr(config.task, 'batch_size') and config.task.batch_size is not None:
                params['task']['batch_size'] = config.task.batch_size
            # Get n_points from task if available (for linear tasks)
            if hasattr(config.task, 'n_points') and config.task.n_points is not None:
                params['task']['n_points'] = config.task.n_points
            # Clean up None values from task
            params['task'] = {k: v for k, v in params['task'].items() if v is not None}
        
        # General parameters
        params['general'] = {
            'seed': getattr(config, 'seed', None),
            'device': getattr(config, 'device', None),
        }
        params['general'] = {k: v for k, v in params['general'].items() if v is not None}
        
        return params
    
    def index_experiment(self, exp_path: str, task_name: str = "latent") -> bool:
        """
        Index a single experiment by reading its config.json.
        
        Args:
            exp_path: Path to experiment directory (e.g., "results/latent/train_xxx")
            task_name: Task name (e.g., "latent", "linear")
        
        Returns:
            True if successful, False otherwise
        """
        exp_path = os.path.abspath(exp_path)
        exp_name = os.path.basename(exp_path)
        config_path = os.path.join(exp_path, "config.json")
        
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            return False
        
        try:
            # Load config
            with open(config_path, 'r') as f:
                config = ConfigDict(json.load(f))
            
            # Generate config hash
            from .basic import get_hash
            config_hash = get_hash(config)
            
            # Extract key parameters
            key_params = self._extract_key_params(config)
            
            # Insert into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert or update experiment
            cursor.execute("""
                INSERT OR REPLACE INTO experiments (exp_name, exp_path, task_name, config_hash, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (exp_name, exp_path, task_name, config_hash))
            
            exp_id = cursor.lastrowid
            if exp_id is None:
                cursor.execute("SELECT id FROM experiments WHERE exp_name = ?", (exp_name,))
                exp_id = cursor.fetchone()[0]
            
            # Delete old parameters
            cursor.execute("DELETE FROM config_params WHERE exp_id = ?", (exp_id,))
            
            # Insert parameters
            for category, params in key_params.items():
                for param_name, param_value in params.items():
                    # Convert value to string for storage
                    value_str = json.dumps(param_value) if isinstance(param_value, (dict, list)) else str(param_value)
                    cursor.execute("""
                        INSERT INTO config_params (exp_id, category, param_name, param_value)
                        VALUES (?, ?, ?, ?)
                    """, (exp_id, category, param_name, value_str))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error indexing {exp_path}: {e}")
            return False
    
    def index_all_experiments(self, root_dir: str = "results", task_names: Optional[List[str]] = None):
        """
        Index all experiments in the results directory.
        
        Args:
            root_dir: Root directory containing task folders (default: "results")
            task_names: List of task names to index (e.g., ["latent", "linear"]).
                       If None, scans all subdirectories.
        """
        if task_names is None:
            # Scan for task directories
            task_names = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))]
        
        total = 0
        success = 0
        
        for task_name in task_names:
            task_dir = os.path.join(root_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            
            print(f"Indexing experiments in {task_dir}...")
            
            for exp_name in os.listdir(task_dir):
                exp_path = os.path.join(task_dir, exp_name)
                if os.path.isdir(exp_path) and exp_name.startswith("train_"):
                    total += 1
                    if self.index_experiment(exp_path, task_name):
                        success += 1
            
        print(f"\nIndexed {success}/{total} experiments successfully.")
    
    def search_experiments(
        self,
        task_name: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for experiments matching given criteria.
        
        Args:
            task_name: Task name filter (e.g., "latent", "linear")
            **kwargs: Parameter filters, e.g., vocab_size=10, emb_dim=128
        
        Returns:
            List of experiment dictionaries with all their parameters
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if task_name:
            conditions.append("e.task_name = ?")
            params.append(task_name)
        
        # Add parameter conditions
        for key, value in kwargs.items():
            # Try to match category.param_name format
            if '.' in key:
                category, param_name = key.split('.', 1)
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM config_params cp
                        WHERE cp.exp_id = e.id
                        AND cp.category = ?
                        AND cp.param_name = ?
                        AND cp.param_value = ?
                    )
                """)
                params.extend([category, param_name, str(value)])
            else:
                # Try common parameter names across categories
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM config_params cp
                        WHERE cp.exp_id = e.id
                        AND cp.param_name = ?
                        AND cp.param_value = ?
                    )
                """)
                params.extend([key, str(value)])
        
        query = """
            SELECT DISTINCT e.* FROM experiments e
        """
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY e.indexed_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get parameters for each experiment
        results = []
        for row in rows:
            exp_id = row['id']
            cursor.execute("""
                SELECT category, param_name, param_value
                FROM config_params
                WHERE exp_id = ?
            """, (exp_id,))
            
            exp_data = {
                'exp_name': row['exp_name'],
                'exp_path': row['exp_path'],
                'task_name': row['task_name'],
                'config_hash': row['config_hash'],
                'indexed_at': row['indexed_at'],
                'params': {}
            }
            
            for cat, pname, pvalue in cursor.fetchall():
                if cat not in exp_data['params']:
                    exp_data['params'][cat] = {}
                try:
                    exp_data['params'][cat][pname] = json.loads(pvalue)
                except:
                    exp_data['params'][cat][pname] = pvalue
            
            results.append(exp_data)
        
        conn.close()
        return results
    
    def export_to_json(self, output_path: str = "results/experiment_index.json"):
        """Export all experiments to JSON file for web UI."""
        experiments = self.search_experiments()
        
        output = {
            'exported_at': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': experiments
        }
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Exported {len(experiments)} experiments to {output_path}")
        return output_path


def index_all_experiments(root_dir: str = "results"):
    """
    Convenience function to index all experiments.
    
    Args:
        root_dir: Root directory containing task folders
    """
    indexer = ExperimentIndex()
    indexer.index_all_experiments(root_dir=root_dir)
    indexer.export_to_json()


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    index_all_experiments(root_dir=root)

