__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

PACKAGE_SCAN_CACHE = {}


def remove_package_cache():
    """Remove the package cache that contains temporary store for scanned files."""
    global PACKAGE_SCAN_CACHE
    PACKAGE_SCAN_CACHE = {}
