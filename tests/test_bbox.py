"""Tests for `cartesio.bbox` subpackage and modules."""

import unittest
import numpy as np
import cartesio as cs


class TestCartesioBBox(unittest.TestCase):
    """Tests for `cartesio.bbox` subpackage and modules."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_bbox_area(self):
        d = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(cs.bbox.area(d), 1)
