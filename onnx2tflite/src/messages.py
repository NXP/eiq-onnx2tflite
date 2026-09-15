#
# Copyright 2023 Martin Pavella
# Copyright 2023, 2026 NXP
#
# License: MIT
# See the LICENSE_MIT for more details.
#
"""messages

Module holds console output styling (`Style`) and reusable CLI hint strings (`Message`).
"""


class Style:
    """Strings used to set a color and other styles to the output printed to console.

    example usage:
        logger.warning(
            f'{messages.Style.orange + messages.Style.bold}Some warning. {messages.Style.end}Additional info.'
        )

    """

    red = "\033[91m"
    green = "\033[92m"
    orange = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    cyan = "\033[96m"

    bold = "\033[1m"
    underline = "\033[4m"

    end = "\033[0m"


class Message:
    """Custom messages, that are printed to console from different locations in the code."""

    ALLOW_SELECT_OPS = (
        "If you want to convert the model using the SELECT_TF_OPS, run the conversion again with "
        f"the flag {Style.bold + Style.cyan}--allow-select-ops{Style.end}."
    )

    GUARANTEE_NON_NEGATIVE_INDICES = (
        f"{Style.green}If you know that the indices are always non-negative, you can run"
        f" the converter with the flag {Style.bold + Style.cyan}--guarantee-non-negative-indices"
        f"{Style.end}."
    )

    CAST_INT64_TO_INT32 = (
        f"Use option {Style.bold + Style.cyan}--cast-int64-to-int32{Style.end} to disable this "
        "check and re-cast input/output to INT32."
    )

    IGNORE_OPSET_VERSION = (
        "If you want to try and convert the model anyway, run the conversion again with the flag "
        f"{Style.bold + Style.cyan}--skip-opset-version-check{Style.end}. Keep in mind that the output"
        " TFLite model may potentially be invalid."
    )
