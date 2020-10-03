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

    def test_nms(self):
        bbs = np.array(
            [
                [0, 0, 100, 100],
                [0, 0, 90, 90],
            ]
        )
        iou = cs.bbox.iou_single(bbs[0], bbs[1])
        keep = cs.bbox.nms(bbs, thresh=iou - 0.0001)
        self.assertArrayEqual(keep, np.array([0], dtype=np.int32), type_strict=True)

        keep = cs.bbox.nms(bbs, thresh=iou + 0.0001)
        self.assertArrayEqual(
            keep,
            np.array(
                [0, 1],
                dtype=np.int32,
            ),
            type_strict=True,
        )

    def test_nms_with_score(self):
        bbs = np.array(
            [
                [0, 0, 100, 100, 10],
                [0, 0, 90, 90, 20],
            ]
        )
        iou = cs.bbox.iou_single(bbs[0, :4], bbs[1, :4])
        keep = cs.bbox.nms(bbs, thresh=iou - 0.0001)
        self.assertArrayEqual(keep, np.array([1], dtype=np.int32), type_strict=True)

        keep = cs.bbox.nms(bbs, thresh=iou + 0.0001)
        self.assertArrayEqual(
            keep,
            np.array(
                [1, 0],
                dtype=np.int32,
            ),
            type_strict=True,
        )
