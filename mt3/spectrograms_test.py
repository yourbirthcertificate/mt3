from unittest import mock
from absl.testing import absltest
import tensorflow as tf

from mt3 import spectral_ops
from mt3 import spectrograms


class SpectrogramsTest(absltest.TestCase):

  def test_preemphasis_applied(self):
    samples = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    cfg = spectrograms.SpectrogramConfig(
        sample_rate=1,
        hop_width=1,
        num_mel_bins=1,
        preemphasis=0.5)

    with mock.patch.object(spectral_ops, 'compute_logmel',
                           return_value=tf.zeros([1, 1], tf.float32)) as mock_fn:
      spectrograms.compute_spectrogram(samples, cfg)
      mock_fn.assert_called_once()
      called_samples = mock_fn.call_args.args[0]
      expected = tf.constant([1.0, 2.0 - 0.5 * 1.0, 3.0 - 0.5 * 2.0],
                             dtype=tf.float32)
      tf.debugging.assert_near(expected, called_samples)


if __name__ == '__main__':
  absltest.main()
