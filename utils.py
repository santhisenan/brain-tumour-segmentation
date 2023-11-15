import os


def get_file_row(path):
    """Produces ID of a patient, image and mask filenames from a particular path"""
    path_no_ext, ext = os.path.splitext(path)
    filename = os.path.basename(path)

    patient_id = "_".join(
        filename.split("_")[:3]
    )  # Patient ID in the csv file consists of 3 first filename segments

    return [patient_id, path, f"{path_no_ext}_mask{ext}"]
