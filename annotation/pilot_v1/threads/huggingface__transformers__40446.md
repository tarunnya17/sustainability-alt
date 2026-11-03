# #40446 [pull_request] Add convert_segmentation_map_to_binary_masks_sorted function for hand…
https://github.com/huggingface/transformers/pull/40446  ·  opened by Ahmed-G-ElTaher on 2025-08-26

## [body] Ahmed-G-ElTaher (NONE) · opening post · 2025-08-26
## Pull Request: Add Support for Handling Overlapping Annotations in Mask2Former

### Problem

The current image processing pipeline for Mask2Former doesn't handle overlapping annotations correctly. When annotations overlap, the processing order is arbitrary, causing larger objects to sometimes be overwritten by smaller ones. This can lead to information loss and incorrect segmentation masks.

### Solution

This PR introduces a new function, `convert_segmentation_map_to_binary_masks_sorted`, that addresses this issue by processing object instances in descending order of their area. This ensures that smaller objects consistently appear on top of larger objects, preserving all annotation information in overlapping regions.

The implementation draws inspiration from the approach used in the `coco2masks` function, which sorts annotations by area before processing them. This enhancement is particularly beneficial for datasets with significant object overlaps.

### Changes

A new function `convert_segmentation_map_to_binary_masks_sorted` has been added with the following features:

- Accepts an optional `sort_by_area` parameter (default: `True`).
- Calculates the area of each instance when sorting is enabled.
- Processes instances in descending order of their area.

This new function can serve as a direct replacement for the existing conversion function, providing improved handling of overlapping regions.

### Testing

- Tested with sample COCO annotations containing overlapping objects.
- Verified that all objects are correctly represented in the output masks.
- Confirmed compatibility with existing Mask2Former models.

### Performance Impact

While the area calculation introduces a minor computational overhead, its impact is negligible within the overall processing pipeline. This step occurs during preprocessing rather than during model inference.

## [c3291737184] Rocketknight1 (MEMBER) · issue_comment · 2025-09-15
cc @qubvel @NielsRogge

---
outcome: state=open, merged=False
