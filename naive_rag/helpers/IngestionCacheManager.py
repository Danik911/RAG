from datetime import datetime

from llama_index.core.ingestion import IngestionCache


class SmartIngestionCache:
    def __init__(self, max_entries=1000, max_age_days=30):
        self.cache = IngestionCache(collection='rag_articles')
        self.max_entries = max_entries
        self.max_age_days = max_age_days

    def prune_old_entries(self):
        # Remove entries older than specified days
        current_time = datetime.now()
        nodes = self.cache.get_nodes()
        fresh_nodes = [
            node for node in nodes
            if (current_time - node.metadata.get('ingestion_date', current_time)).days <= self.max_age_days
        ]
        self.cache.put_nodes(fresh_nodes)

    def enforce_max_entries(self):
        # Limit total number of cached entries
        nodes = self.cache.get_nodes()
        if len(nodes) > self.max_entries:
            # Keep most recent entries
            nodes = sorted(nodes, key=lambda x: x.metadata.get('ingestion_date', datetime.min))[:self.max_entries]
            self.cache.put_nodes(nodes)

    def get_cache(self):
        # Return the underlying IngestionCache
        return self.cache
