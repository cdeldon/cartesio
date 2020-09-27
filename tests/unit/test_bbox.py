"""Tests for `cartesio.bbox` subpackage and modules."""
import numpy as np

import cartesio as cs
from .utils import TestCase


class TestCartesioBBox(TestCase):
    """Tests for `cartesio.bbox` subpackage and modules."""

    def test_area(self):
        bb = np.array([0, 0, 0, 0])
        self.assertEqual(cs.bbox.area(bb), 0)

        bb = np.array([0, 0, 0, 1])
        self.assertEqual(cs.bbox.area(bb), 0)

        bb = np.array([0, 0, 1, 0])
        self.assertEqual(cs.bbox.area(bb), 0)

        bb = np.array([0, 0, 1, 1])
        self.assertEqual(cs.bbox.area(bb), 1)

        bb = np.array([1, 1, 2, 2])
        self.assertEqual(cs.bbox.area(bb), 1)

        bb = np.array([1.5, 1.5, 2.0, 3.5])
        self.assertEqual(cs.bbox.area(bb), 1)

    def test_iou_single(self):
        bb_0 = np.array([0, 0, 1, 1])
        bb_1 = np.array([0, 0, 1, 1])
        self.assertEqual(cs.bbox.iou_single(bb_0, bb_1), 1)

        bb_0 = np.array([0, 0, 1, 1])
        bb_1 = np.array([0, 0, 2, 2])
        self.assertEqual(cs.bbox.iou_single(bb_0, bb_1), 1 / 4)

        bb_0 = np.array([1, 1, 2, 2])
        bb_1 = np.array([0, 0, 2, 2])
        self.assertEqual(cs.bbox.iou_single(bb_0, bb_1), 1 / 4)

        bb_0 = np.array([1.5, 1.5, 2.5, 2.5])
        bb_1 = np.array([0, 0, 2, 2])
        self.assertEqual(cs.bbox.iou_single(bb_0, bb_1), 0.5 ** 2 / (5 - 0.5 ** 2))

    def test_iou(self):
        bbs_0 = np.array([[0, 0, 1, 1]])
        bbs_1 = np.array([[0, 0, 1, 1]])

        ious = cs.bbox.iou(bbs_0, bbs_1)
        self.assertArrayEqual(
            ious, np.array([[1.0]], dtype=np.float32), type_strict=True
        )

        bbs_0 = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        bbs_1 = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])

        ious = cs.bbox.iou(bbs_0, bbs_1)
        self.assertArrayEqual(
            ious,
            np.array(
                [
                    [1.0, 0.25],
                    [0.25, 1.0],
                ],
                dtype=np.float32,
            ),
            type_strict=True,
        )

        bbs_0 = np.array([[0, 0, 1, 1], [0, 0, 2, 2]])
        bbs_1 = np.array([[0, 0, 1, 2]])

        ious = cs.bbox.iou(bbs_0, bbs_1)
        self.assertArrayEqual(
            ious,
            np.array(
                [
                    [0.5],
                    [0.5],
                ],
                dtype=np.float32,
            ),
            type_strict=True,
        )
