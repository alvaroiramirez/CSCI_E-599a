from datetime import datetime
from PIL.ExifTags import TAGS

CATEGORY_MAP = {
    "Affected_Building": 1,
    "Major_Damage": 2
}



def get_date_captured(image, verbose=False):
    """
    Extracts date from EXIF if available. Tries .getexif() first, then _getexif().
    """
    try:
        # Try the modern EXIF method first
        if hasattr(image, "getexif"):
            exif = image.getexif()
        else:
            exif = image._getexif()

        if exif:
            preferred_tags = ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]
            for tag, val in exif.items():
                tag_name = TAGS.get(tag, str(tag))
                if verbose:
                    print(f"EXIF → {tag_name}: {val}")
                if tag_name in preferred_tags:
                    try:
                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S").isoformat()
                    except Exception as e:
                        if verbose:
                            print(f"⛔ Failed parsing {tag_name}: {e}")
    except Exception as e:
        if verbose:
            print(f"⚠️ EXIF read error: {e}")

    if verbose:
        print("📆 No valid EXIF date found, using current timestamp.")
    return datetime.now().isoformat()
