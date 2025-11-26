"""
Lazy import manager for BioMedStatX
Improves startup time by loading heavy modules only when needed
"""

import sys
from typing import Any, Dict, Optional

class LazyImportManager:
    """Manages lazy loading of heavy modules to improve startup performance"""
    
    def __init__(self):
        self._modules: Dict[str, Any] = {}
        self._import_errors: Dict[str, Exception] = {}
    
    def get_module(self, module_name: str, import_path: str, alias: Optional[str] = None) -> Any:
        """
        Lazily import a module and cache it
        
        Args:
            module_name: Key to store the module under
            import_path: The import path (e.g., 'scipy.stats')
            alias: Optional alias for the module
            
        Returns:
            The imported module
            
        Raises:
            ImportError: If the module cannot be imported
        """
        if module_name in self._modules:
            return self._modules[module_name]
            
        if module_name in self._import_errors:
            raise self._import_errors[module_name]
            
        try:
            # Dynamic import
            module = __import__(import_path, fromlist=[''])
            if alias:
                # For cases like 'import pandas as pd'
                self._modules[alias] = module
            self._modules[module_name] = module
            return module
        except ImportError as e:
            self._import_errors[module_name] = e
            raise e
    
    def get_pingouin(self):
        """Get pingouin module (pg)"""
        return self.get_module('pingouin', 'pingouin', 'pg')
    
    def get_scipy_stats(self):
        """Get scipy.stats module"""
        return self.get_module('scipy_stats', 'scipy.stats')
    
    def get_seaborn(self):
        """Get seaborn module"""
        return self.get_module('seaborn', 'seaborn', 'sns')
    
    def is_available(self, module_name: str) -> bool:
        """Check if a module is available without importing it"""
        if module_name in self._modules:
            return True
        if module_name in self._import_errors:
            return False
        
        try:
            self.get_module(module_name, module_name)
            return True
        except ImportError:
            return False
    
    def clear_cache(self):
        """Clear the module cache (useful for testing)"""
        self._modules.clear()
        self._import_errors.clear()

# Global instance
lazy_imports = LazyImportManager()

# Convenience functions for common modules
def get_pingouin():
    """Get pingouin as pg"""
    return lazy_imports.get_pingouin()

def get_scipy_stats():
    """Get scipy.stats"""
    return lazy_imports.get_scipy_stats()

def get_seaborn():
    """Get seaborn as sns"""
    return lazy_imports.get_seaborn()

# Additional convenience functions for heavy modules
def get_statsmodels_multicomp():
    """Get statsmodels multicomp module"""
    return lazy_imports.get_module('statsmodels_multicomp', 'statsmodels.stats.multicomp')

def get_xlsxwriter():
    """Get xlsxwriter module"""
    return lazy_imports.get_module('xlsxwriter', 'xlsxwriter')

def get_matplotlib_pyplot():
    """Get matplotlib.pyplot"""
    return lazy_imports.get_module('matplotlib_pyplot', 'matplotlib.pyplot', 'plt')

def get_scikit_posthocs():
    """Get scikit_posthocs module"""
    return lazy_imports.get_module('scikit_posthocs', 'scikit_posthocs')

def get_statsmodels_multitest():
    """Get statsmodels multitest multipletests function with fallback"""
    try:
        from statsmodels.stats.multitest import multipletests
        return multipletests
    except ImportError:
        # Fallback implementation using Bonferroni correction
        def multipletests_fallback(pvals, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False):
            """
            Fallback implementation of multipletests using simple Bonferroni correction
            """
            import numpy as np
            
            pvals = np.asarray(pvals)
            if len(pvals) == 0:
                return np.array([]), np.array([]), 0.0, 0.0
            
            # Simple Bonferroni correction
            alphacBonf = alpha / len(pvals)
            reject = pvals <= alphacBonf
            pvals_corrected = np.minimum(pvals * len(pvals), 1.0)
            
            # Return format: (reject, pvals_corrected, alphacSidak, alphacBonf)
            return reject, pvals_corrected, alpha, alphacBonf
        
        print("\n" + "="*80)
        print("⚠️  WARNUNG: STATSMODELS NICHT VERFÜGBAR!")
        print("="*80)
        print("Das Programm verwendet jetzt einen einfachen Bonferroni-Fallback.")
        print("Für bessere statistische Korrekturen installiere statsmodels:")
        print("   pip install statsmodels")
        print("="*80 + "\n")
        return multipletests_fallback

# Pre-populate commonly used imports at startup if desired
def preload_critical_modules():
    """Preload modules that are almost always needed"""
    try:
        # Only preload if they're in requirements and commonly used
        lazy_imports.get_module('numpy', 'numpy', 'np')
        lazy_imports.get_module('pandas', 'pandas', 'pd')
        # matplotlib and heavy modules will be loaded on demand
    except ImportError:
        pass  # Fail silently for optional preloading
