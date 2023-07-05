from preprocess import MinMaxNormaliser
import librosa

class SoundGenerator:

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):

            log_spectrogram = spectrogram[:, :, 0]

            denorm_log_spec = self._min_max_normaliser.denormalise(log_spectrogram, min_max_value["min"], min_max_value["max"])

            spec = librosa.db_to_amplitude(denorm_log_spec)

            signal = librosa.istft(spec, hop_length=self.hop_length)

            signals.append(signal)

        return signals
