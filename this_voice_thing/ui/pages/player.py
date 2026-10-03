"""The output list and audio player under the Generate page."""

import os

from PySide6.QtCore import QUrl
from PySide6.QtCore import Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QCheckBox
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtWidgets import QListWidget
from PySide6.QtWidgets import QListWidgetItem
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtWidgets import QSlider

from this_voice_thing.core import audio_effects


class Player:
    """The output list and audio player under the Generate page. Mixed into ChatterboxApp."""

    def _build_player_card(self):
        player_card, player_layout = self._make_card()
        player_header = QHBoxLayout()
        player_title = QLabel("Player")
        player_title.setObjectName("CardTitle")
        player_header.addWidget(player_title)
        player_header.addSpacing(12)
        self.autoplay_checkbox = QCheckBox("Auto-play results")
        self.autoplay_checkbox.setChecked(True)
        player_header.addWidget(self.autoplay_checkbox)
        player_header.addStretch(1)
        self.current_file_label = QLabel("Currently playing: None")
        self.current_file_label.setObjectName("Muted")
        player_header.addWidget(self.current_file_label)
        player_layout.addLayout(player_header)
        player_controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.play_pause_button.setEnabled(False)
        self.play_pause_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.play_pause_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_audio)
        self.stop_button.setEnabled(False)
        self.stop_button.setMinimumWidth(80)
        player_controls_layout.addWidget(self.stop_button)
        self.current_time_label = QLabel("00:00")
        self.playhead_slider = QSlider(Qt.Orientation.Horizontal)
        self.playhead_slider.sliderPressed.connect(self.slider_pressed)
        self.playhead_slider.sliderMoved.connect(self.seek_audio_on_move)
        self.playhead_slider.sliderReleased.connect(self.slider_released)
        self.playhead_slider.setEnabled(False)
        self.duration_label = QLabel("00:00")
        player_controls_layout.addSpacing(8)
        player_controls_layout.addWidget(self.current_time_label)
        player_controls_layout.addWidget(self.playhead_slider, 1)
        player_controls_layout.addWidget(self.duration_label)
        player_layout.addLayout(player_controls_layout)
        history_label = QLabel("Generated files (double-click to play)")
        history_label.setObjectName("Muted")
        player_layout.addWidget(history_label)
        self.output_log_listwidget = QListWidget()
        self.output_log_listwidget.itemDoubleClicked.connect(
            self.play_selected_from_log)
        self.output_log_listwidget.setMinimumHeight(70)
        self.output_log_listwidget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        player_layout.addWidget(self.output_log_listwidget, 1)
        return player_card

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
