"""The output list and audio player under the Generate page."""

import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QListWidgetItem, QMessageBox

from this_voice_thing.core import audio_effects


class Player:
    """The output list and audio player under the Generate page. Mixed into ChatterboxApp."""

    def update_output_log(self):
        self.output_log_listwidget.clear()
        try:
            files = sorted([os.path.join(self.output_directory, f)for f in os.listdir(
                self.output_directory)if f.lower().endswith(audio_effects.AUDIO_EXTENSIONS)], key=os.path.getmtime, reverse=True)
            for f_path in files:
                item = QListWidgetItem(os.path.basename(f_path))
                item.setData(Qt.ItemDataRole.UserRole, f_path)
                self.output_log_listwidget.addItem(item)
        except Exception as e:
            print(f"Error updating output log: {e}")

    def play_selected_from_log(self, item: QListWidgetItem):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and os.path.exists(file_path):
            self.current_audio_file = file_path
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.current_file_label.setText(
                f"Playing from log: {os.path.basename(file_path)}")
            self.play_pause_button.setText("Pause")
            self.media_player.play()
            self.play_pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.playhead_slider.setEnabled(True)
        else:
            QMessageBox.warning(self, "File Error",
                                "Could not find or play selected audio file.")
            self.update_output_log()

    def toggle_play_pause(self):
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            # Store current position before pausing
            self.paused_position = self.media_player.position()
            self.media_player.pause()
            # self.play_pause_button.setText("Play") # Done by handle_playback_state_changed
        else:  # Was Paused or Stopped
            if not self.current_audio_file or not os.path.exists(self.current_audio_file):
                QMessageBox.warning(
                    self, "No Audio", "No audio file loaded to play.")
                return

            # Ensure media source is set
            if self.media_player.source().isEmpty() or \
               self.media_player.source().toLocalFile() != self.current_audio_file:
                self.media_player.setSource(
                    QUrl.fromLocalFile(self.current_audio_file))
                # If source changed or was empty, assume play from start or last paused pos
                if hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                    self.media_player.setPosition(self.paused_position)
                else:  # Play from start if no paused_position or at end of media
                    self.media_player.setPosition(0)
                    self.paused_position = 0  # Reset paused position

            # If resuming from a paused state (and not end of media)
            elif hasattr(self, 'paused_position') and self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # Only set position if it's significantly different (avoids tiny jumps if already paused)
                # or if it was explicitly stopped and then play is hit again.
                if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                    self.media_player.setPosition(self.paused_position)

            # If it was fully stopped (and not at end of media), or if no paused_position, play from current slider or 0
            elif self.media_player.playbackState() == QMediaPlayer.PlaybackState.StoppedState and \
                    self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia:
                # If stopped, it might have a valid position already (e.g. user seeked then stopped)
                # If no paused_position exists, it implies fresh play or play after stop
                if not hasattr(self, 'paused_position'):
                    self.media_player.setPosition(0)  # Default to start
                # else: play from current position (which might be 0 if stopped at start)

            self.media_player.play()
            # self.play_pause_button.setText("Pause") # Done by handle_playback_state_changed

    def stop_audio(self):
        self.media_player.stop()
        self.play_pause_button.setText("Play")

    # --- Slider Seeking Logic ---
    def slider_pressed(self): self.is_seeking_audio = True

    def seek_audio_on_move(self, position):
        if self.is_seeking_audio:
            self.media_player.setPosition(position)
            self.current_time_label.setText(self.format_time(position))

    def slider_released(self):
        if self.is_seeking_audio:
            self.is_seeking_audio = False
            self.media_player.setPosition(self.playhead_slider.value())

    def update_slider_position(self, position):
        if not self.is_seeking_audio:
            self.playhead_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    def update_duration_info(self, duration):
        if duration > 0:  # Only set range if duration is valid
            self.playhead_slider.setRange(0, duration)
            self.duration_label.setText(self.format_time(duration))
        else:  # Reset if duration is 0 or invalid (e.g. after stop or error)
            self.playhead_slider.setRange(0, 0)
            self.duration_label.setText("00:00")
            self.current_time_label.setText("00:00")

    def format_time(self, ms: int) -> str:  # Your improved version
        if ms < 0:
            ms = 0
        total_seconds_val = ms // 1000
        minutes_val = total_seconds_val // 60
        seconds_remainder_val = total_seconds_val % 60
        return f"{minutes_val:02}:{seconds_remainder_val:02}"

    def handle_playback_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("Pause")
        else:
            self.play_pause_button.setText("Play")
        # This logic might conflict with explicit stop, if positionChanged handle slider reset mainly
        # if state==QMediaPlayer.PlaybackState.StoppedState and self.media_player.position()==0:   # Reset slider and time if stopped at start
        #     if self.media_player.mediaStatus()==QMediaPlayer.MediaStatus.EndOfMedia: # End of media reached
        #         self.playhead_slider.setValue(0);self.current_time_label.setText("00:00") # Reset to start

    def handle_media_error(self):
        error_string = self.media_player.errorString()
        if error_string:
            QMessageBox.warning(self, "Media Player Error",
                                f"Error playing audio: {error_string}")
        self.set_status_message("Status: Media player error.")
        self.play_pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.playhead_slider.setEnabled(False)
        self.update_duration_info(0)  # Reset duration display on error
