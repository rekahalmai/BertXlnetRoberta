import os
import sys
import torch


def create_reports_directory(params):
    """
    Verifies whether there exist already a REPORTS_DIR and if not, it creates one.
    :return: None
    """

    task_name = params["task_name"]
    reports_dir = f"reports/{task_name}_evaluation_report/"

    if os.path.exists(reports_dir) and os.listdir(reports_dir):
        reports_dir += f"/report_{len(os.listdir(reports_dir))}"
        os.makedirs(reports_dir)

    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        reports_dir += f"/report_{len(os.listdir(reports_dir))}"
        os.makedirs(reports_dir)


def create_output_directory(params):
    """
    If output directory does not yet exist, creates one.
    If it already exists, it does not do anything.

    :return: None
    """

    output_dir = params["output_dir"]

    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(ValueError(f"Output directory ({output_dir}) already exists and is not empty."))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def main():
    # need to load and drop it with torch as the dict includes the device
    # ("device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = torch.load(sys.argv[1])
    create_reports_directory(params)
    create_output_directory(params)


if __name__ == "__main__":
    main()

