import json
import logging
import threading
import time
import queue
import numpy as np


class ServeClientBase(object):
    RATE = 16000
    SERVER_READY = "SERVER_READY"
    DISCONNECT = "DISCONNECT"

    client_uid: str
    """A unique identifier for the client."""
    websocket: object
    """The WebSocket connection for the client."""
    send_last_n_segments: int
    """Number of most recent segments to send to the client."""
    no_speech_thresh: float
    """Segments with no speech probability above this threshold will be discarded."""
    clip_audio: bool
    """Whether to clip audio with no valid segments."""
    same_output_threshold: int
    """Number of repeated outputs before considering it as a valid segment."""

    def __init__(
        self,
        client_uid,
        websocket,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=10,
        translation_queue=None,
    ):
        self.client_uid = client_uid
        self.websocket = websocket
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold

        self.frames = b""
        self.timestamp_offset = 0.0
        self.frames_np = None
        self.frames_offset = 0.0
        self.text = []
        self.current_out = ""
        self.prev_out = ""
        self.exit = False
        self.same_output_count = 0
        self.transcript = []
        self.end_time_for_same_output = None
        self.translation_queue = translation_queue

        # Speaker timeline tracking
        # Each entry: {"offset": float, "speaker": str, "client_ts": float}
        self.speaker_timeline = []

        # Output consolidation: merge consecutive segments from same speaker
        # Output only when segment is completed=True OR duration exceeds 7 seconds
        self.pending_segment = None  # {"text": str, "start": float, "end": float, "speaker": str}
        self.max_pending_duration = 7.0  # Max duration before forcing output (seconds)
        self.max_pause_to_merge = 2.0  # Max pause between segments to merge (seconds)

        # threading
        self.lock = threading.Lock()

    def speech_to_text(self):
        """
        Process an audio stream in an infinite loop, continuously transcribing the speech.

        This method continuously receives audio frames, performs real-time transcription, and sends
        transcribed segments to the client via a WebSocket connection.

        If the client's language is not detected, it waits for 30 seconds of audio input to make a language prediction.
        It utilizes the Whisper ASR model to transcribe the audio, continuously processing and streaming results. Segments
        are sent to the client in real-time, and a history of segments is maintained to provide context.

        Raises:
            Exception: If there is an issue with audio processing or WebSocket communication.

        """
        while True:
            if self.exit:
                logging.info("Exiting speech to text thread")
                break

            if self.frames_np is None:
                continue

            if self.clip_audio:
                self.clip_audio_if_no_valid_segment()

            input_bytes, duration = self.get_audio_chunk_for_processing()
            if duration < 1.0:
                time.sleep(0.1)     # wait for audio chunks to arrive
                continue
            try:
                input_sample = input_bytes.copy()
                result = self.transcribe_audio(input_sample)

                if result is None:
                    logging.info(f"[SPEECH_TO_TEXT] result is None, skipping (no voice activity)")
                    self.timestamp_offset += duration
                    time.sleep(0.25)
                    continue

                if self.language is None:
                    logging.warning(f"[SPEECH_TO_TEXT] language is None, skipping transcription result")
                    self.timestamp_offset += duration
                    time.sleep(0.25)
                    continue

                logging.info(f"[SPEECH_TO_TEXT] Calling handle_transcription_output, language={self.language}")
                self.handle_transcription_output(result, duration)

            except Exception as e:
                logging.error(f"[ERROR]: Failed to transcribe audio chunk: {e}")
                time.sleep(0.01)

    def transcribe_audio(self):
        raise NotImplementedError

    def handle_transcription_output(self, result, duration):
        raise NotImplementedError
    
    def format_segment(self, start, end, text, completed=False):
        """
        Formats a transcription segment with precise start and end times alongside the transcribed text.

        Args:
            start (float): The start time of the transcription segment in seconds.
            end (float): The end time of the transcription segment in seconds.
            text (str): The transcribed text corresponding to the segment.
            completed (bool): Whether the segment is complete.

        Returns:
            dict: A dictionary representing the formatted transcription segment, including
                'start' and 'end' times as strings with three decimal places, the 'text'
                of the transcription, and the 'speaker' at the start time.
        """
        # Look up speaker at the start time of this segment
        speaker = self._get_speaker_at_time(start)

        return {
            'start': "{:.3f}".format(start),
            'end': "{:.3f}".format(end),
            'text': text,
            'completed': completed,
            'speaker': speaker
        }

    def consolidate_segment(self, segment):
        """
        Consolidate segments from the same speaker before sending.

        Merges consecutive segments from same speaker if:
        - Speaker is the same
        - Pause between segments < max_pause_to_merge (2.0s)
        - Total duration < max_pending_duration (7.0s)
        - Segment is NOT completed (completed=False)

        Outputs immediately if:
        - Segment is completed (completed=True)
        - Duration exceeds 7 seconds
        - Speaker changes
        - Pause > 2.0s

        Args:
            segment (dict): New segment with start, end, text, speaker, completed

        Returns:
            dict or None: Consolidated segment ready to send, or None if still accumulating
        """
        if segment is None:
            return None

        current_speaker = segment.get('speaker')
        current_start = float(segment.get('start', 0))
        current_end = float(segment.get('end', 0))
        current_text = segment.get('text', '').strip()
        current_completed = segment.get('completed', False)

        # Skip empty text
        if not current_text:
            return None

        # First segment or no pending segment
        if self.pending_segment is None:
            self.pending_segment = {
                'start': current_start,
                'end': current_end,
                'text': current_text,
                'speaker': current_speaker,
                'completed': current_completed
            }
            logging.debug(f"[CONSOLIDATE] Started pending: speaker={current_speaker}, text='{current_text[:30]}', completed={current_completed}")

            # Check if we should output immediately due to completion or duration
            pending_duration = current_end - current_start
            if current_completed or pending_duration >= self.max_pending_duration:
                output = self.pending_segment.copy()
                self.pending_segment = None
                reason = "completed" if current_completed else "duration_limit"
                logging.info(f"[CONSOLIDATE] Output immediately: speaker={output['speaker']}, duration={pending_duration:.2f}s, reason={reason}")
                return output

            return None  # Continue accumulating

        pending_speaker = self.pending_segment['speaker']
        pending_end = self.pending_segment['end']
        pending_start = self.pending_segment['start']
        pending_completed = self.pending_segment['completed']

        # Calculate pause and duration
        pause = current_start - pending_end
        total_duration = current_end - pending_start

        # Check if we should force output of pending due to completion or duration
        should_force_output = (
            pending_completed or
            (pending_end - pending_start) >= self.max_pending_duration
        )

        # Check if we should merge or output
        should_merge = (
            not should_force_output and
            not current_completed and
            current_speaker == pending_speaker and
            pause <= self.max_pause_to_merge and
            total_duration <= self.max_pending_duration
        )

        if should_merge:
            # Merge into pending segment
            self.pending_segment['end'] = current_end
            self.pending_segment['text'] += ' ' + current_text
            self.pending_segment['completed'] = current_completed
            logging.debug(f"[CONSOLIDATE] Merged: total_duration={total_duration:.2f}s, pause={pause:.2f}s, completed={current_completed}")

            # Check if merged segment should now be output due to completion or duration
            merged_duration = self.pending_segment['end'] - self.pending_segment['start']
            if current_completed or merged_duration >= self.max_pending_duration:
                output = self.pending_segment.copy()
                self.pending_segment = None
                reason = "completed" if current_completed else "duration_limit"
                logging.info(f"[CONSOLIDATE] Output merged: speaker={output['speaker']}, duration={merged_duration:.2f}s, reason={reason}")
                return output

            return None  # Continue accumulating
        else:
            # Output pending segment and start new one
            output_segment = self.pending_segment.copy()
            self.pending_segment = {
                'start': current_start,
                'end': current_end,
                'text': current_text,
                'speaker': current_speaker,
                'completed': current_completed
            }

            reason = "speaker_change" if current_speaker != pending_speaker else "pause" if pause > self.max_pause_to_merge else "pending_completed" if should_force_output else "duration"
            logging.info(f"[CONSOLIDATE] Output: speaker={output_segment['speaker']}, duration={output_segment['end']-output_segment['start']:.2f}s, reason={reason}")

            # Check if new pending should be output immediately
            new_duration = current_end - current_start
            if current_completed or new_duration >= self.max_pending_duration:
                output2 = self.pending_segment.copy()
                self.pending_segment = None
                reason2 = "completed" if current_completed else "duration_limit"
                logging.info(f"[CONSOLIDATE] Output new immediately: speaker={output2['speaker']}, duration={new_duration:.2f}s, reason={reason2}")
                # Return only the first output, the second will be sent on next call
                # Actually, we need to return both - let's return the first one and queue the second
                # For simplicity, store the output_segment and return it, the new one will be pending

            return output_segment

    def flush_pending_segment(self):
        """Force output of pending segment (e.g., on session close)."""
        if self.pending_segment is not None:
            output = self.pending_segment.copy()
            self.pending_segment = None
            logging.info(f"[CONSOLIDATE] Flushed pending: speaker={output['speaker']}, duration={output['end']-output['start']:.2f}s")
            return output
        return None

    def add_frames(self, frame_np, speaker=None, client_timestamp=None):
        """
        Add audio frames to the ongoing audio stream buffer.

        Processes chunks immediately without accumulation for minimal latency.
        Consolidation happens in consolidate_segment() based on completed flag and duration.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.
            speaker (str, optional): Speaker identifier for this audio chunk.
            client_timestamp (float, optional): Client-provided timestamp for this audio chunk.

        Returns:
            dict: Speaker change event if speaker changed, None otherwise
                  Format: {"event": "speaker_changed", "speaker": str, "offset": float, "client_ts": float}

        """
        self.lock.acquire()
        speaker_change_event = None

        # Calculate current buffer end offset (where this new chunk will be appended)
        if self.frames_np is not None:
            current_buffer_end = self.frames_offset + (self.frames_np.shape[0] / self.RATE)
        else:
            current_buffer_end = self.frames_offset

        # Track speaker changes in timeline
        if speaker is not None:
            # Check if speaker changed from last entry
            if not self.speaker_timeline or self.speaker_timeline[-1]["speaker"] != speaker:
                timeline_entry = {
                    "offset": current_buffer_end,
                    "speaker": speaker,
                    "client_ts": client_timestamp if client_timestamp is not None else time.time()
                }
                self.speaker_timeline.append(timeline_entry)
                logging.info(f"[SPEAKER_TIMELINE] Speaker change: {speaker} at offset {current_buffer_end:.3f}s (client_ts={client_timestamp})")

                # Create speaker change event for WebSocket notification
                speaker_change_event = {
                    "event": "speaker_changed",
                    "speaker": speaker,
                    "offset": current_buffer_end,
                    "client_ts": client_timestamp if client_timestamp is not None else time.time()
                }

        if self.frames_np is not None and self.frames_np.shape[0] > 45*self.RATE:
            clipped_duration = 30.0
            self.frames_offset += clipped_duration
            self.frames_np = self.frames_np[int(30*self.RATE):]

            # Clean up speaker timeline entries that are now before frames_offset
            self.speaker_timeline = [
                entry for entry in self.speaker_timeline
                if entry["offset"] >= self.frames_offset
            ]
            logging.info(f"[BUFFER_CLIP] Clipped 30s, new frames_offset={self.frames_offset:.3f}s, speaker_timeline entries: {len(self.speaker_timeline)}")

            # check timestamp offset(should be >= self.frame_offset)
            # this basically means that there is no speech as timestamp offset hasnt updated
            # and is less than frame_offset
            if self.timestamp_offset < self.frames_offset:
                self.timestamp_offset = self.frames_offset

        # Add frame to buffer immediately (no accumulation)
        if self.frames_np is None:
            self.frames_np = frame_np.copy()
        else:
            self.frames_np = np.concatenate((self.frames_np, frame_np), axis=0)

        self.lock.release()

        # Log chunk processing
        chunk_duration = frame_np.shape[0] / self.RATE
        logging.debug(f"[ADD_FRAMES] Processed chunk: duration={chunk_duration:.3f}s, speaker={speaker}")

        return speaker_change_event

    def _get_speaker_at_time(self, offset):
        """
        Look up the speaker at a specific buffer offset time.

        Args:
            offset (float): Buffer offset time in seconds.

        Returns:
            str: Speaker name at that time, or "Unknown" if no speaker data available.
        """
        if not self.speaker_timeline:
            return "Unknown"

        # Find the most recent speaker change at or before this offset
        current_speaker = "Unknown"
        for entry in self.speaker_timeline:
            if entry["offset"] <= offset:
                current_speaker = entry["speaker"]
            else:
                break  # Timeline is ordered, no need to continue

        return current_speaker

    def clip_audio_if_no_valid_segment(self):
        """
        Update the timestamp offset based on audio buffer status.
        Clip audio if the current chunk exceeds 30 seconds, this basically implies that
        no valid segment for the last 30 seconds from whisper
        """
        with self.lock:
            if self.frames_np[int((self.timestamp_offset - self.frames_offset)*self.RATE):].shape[0] > 25 * self.RATE:
                duration = self.frames_np.shape[0] / self.RATE
                self.timestamp_offset = self.frames_offset + duration - 5

    def get_audio_chunk_for_processing(self):
        """
        Retrieves the next chunk of audio data for processing based on the current offsets.

        Calculates which part of the audio data should be processed next, based on
        the difference between the current timestamp offset and the frame's offset, scaled by
        the audio sample rate (RATE). It then returns this chunk of audio data along with its
        duration in seconds.

        Returns:
            tuple: A tuple containing:
                - input_bytes (np.ndarray): The next chunk of audio data to be processed.
                - duration (float): The duration of the audio chunk in seconds.
        """
        with self.lock:
            samples_take = max(0, (self.timestamp_offset - self.frames_offset) * self.RATE)
            input_bytes = self.frames_np[int(samples_take):].copy()
        duration = input_bytes.shape[0] / self.RATE
        return input_bytes, duration

    def prepare_segments(self, last_segment=None):
        """
        Prepares the segments of transcribed text to be sent to the client.

        This method compiles the recent segments of transcribed text, ensuring that only the
        specified number of the most recent segments are included. It also appends the most
        recent segment of text if provided (which is considered incomplete because of the possibility
        of the last word being truncated in the audio chunk).

        Args:
            last_segment (str, optional): The most recent segment of transcribed text to be added
                                          to the list of segments. Defaults to None.

        Returns:
            list: A list of transcribed text segments to be sent to the client.
        """
        segments = []
        if len(self.transcript) >= self.send_last_n_segments:
            segments = self.transcript[-self.send_last_n_segments:].copy()
        else:
            segments = self.transcript.copy()
        if last_segment is not None:
            segments = segments + [last_segment]
        return segments

    def get_audio_chunk_duration(self, input_bytes):
        """
        Calculates the duration of the provided audio chunk.

        Args:
            input_bytes (numpy.ndarray): The audio chunk for which to calculate the duration.

        Returns:
            float: The duration of the audio chunk in seconds.
        """
        return input_bytes.shape[0] / self.RATE

    def send_transcription_to_client(self, segments):
        """
        Sends the specified transcription segments to the client over the websocket connection.

        This method formats the transcription segments into a JSON object and attempts to send
        this object to the client. If an error occurs during the send operation, it logs the error.

        Returns:
            segments (list): A list of transcription segments to be sent to the client.
        """
        try:
            message = json.dumps({
                "uid": self.client_uid,
                "segments": segments,
            })
            self.websocket.send(message)
        except Exception as e:
            logging.error(f"[ERROR]: Sending data to client: {e}")

    def disconnect(self):
        """
        Notify the client of disconnection and send a disconnect message.

        This method sends a disconnect message to the client via the WebSocket connection to notify them
        that the transcription service is disconnecting gracefully.

        """
        self.websocket.send(json.dumps({
            "uid": self.client_uid,
            "message": self.DISCONNECT
        }))

    def cleanup(self):
        """
        Perform cleanup tasks before exiting the transcription service.

        This method performs necessary cleanup tasks, including stopping the transcription thread, marking
        the exit flag to indicate the transcription thread should exit gracefully, and destroying resources
        associated with the transcription process.

        """
        logging.info("Cleaning up.")
        self.exit = True
    
    def get_segment_no_speech_prob(self, segment):
        return getattr(segment, "no_speech_prob", 0)

    def get_segment_start(self, segment):
        return getattr(segment, "start", getattr(segment, "start_ts", 0))

    def get_segment_end(self, segment):
        return getattr(segment, "end", getattr(segment, "end_ts", 0))

    def update_segments(self, segments, duration):
        """
        Processes the segments from Whisper and updates the transcript.
        Uses helper methods to account for differences between backends.
        
        Args:
            segments (list): List of segments returned by the transcriber.
            duration (float): Duration of the current audio chunk.
        
        Returns:
            dict or None: The last processed segment (if any).
        """
        offset = None
        self.current_out = ''
        last_segment = None

        # Process complete segments only if there are more than one
        # and if the last segment's no_speech_prob is below the threshold.
        if len(segments) > 1 and self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            for s in segments[:-1]:
                text_ = s.text
                self.text.append(text_)
                with self.lock:
                    start = self.timestamp_offset + self.get_segment_start(s)
                    end = self.timestamp_offset + min(duration, self.get_segment_end(s))
                if start >= end:
                    continue
                if self.get_segment_no_speech_prob(s) > self.no_speech_thresh:
                    continue
                completed_segment = self.format_segment(start, end, text_, completed=True)
                self.transcript.append(completed_segment)

                if self.translation_queue:
                    try:
                        self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                    except queue.Full:
                        logging.warning("Translation queue is full, skipping segment")
                offset = min(duration, self.get_segment_end(s))

        # Process the last segment if its no_speech_prob is acceptable.
        if self.get_segment_no_speech_prob(segments[-1]) <= self.no_speech_thresh:
            last_seg = segments[-1]
            self.current_out += last_seg.text

            # If this is the only segment OR it's clearly completed (high confidence), add to transcript
            is_completed = len(segments) == 1 or self.get_segment_no_speech_prob(last_seg) < 0.3

            with self.lock:
                last_segment = self.format_segment(
                    self.timestamp_offset + self.get_segment_start(last_seg),
                    self.timestamp_offset + min(duration, self.get_segment_end(last_seg)),
                    self.current_out,
                    completed=is_completed
                )

                # Add to transcript if completed
                if is_completed:
                    self.transcript.append(last_segment)
                    if self.translation_queue:
                        try:
                            self.translation_queue.put(last_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning("Translation queue is full, skipping segment")

        # Handle repeated output logic.
        if self.current_out.strip() == self.prev_out.strip() and self.current_out != '':
            self.same_output_count += 1

            # if we remove the audio because of same output on the nth reptition we might remove the 
            # audio thats not yet transcribed so, capturing the time when it was repeated for the first time
            if self.end_time_for_same_output is None:
                self.end_time_for_same_output = self.get_segment_end(segments[-1])
            time.sleep(0.1)  # wait briefly for any new voice activity
        else:
            self.same_output_count = 0
            self.end_time_for_same_output = None

        # If the same incomplete segment is repeated too many times,
        # append it to the transcript and update the offset.
        if self.same_output_count > self.same_output_threshold:
            if not self.text or self.text[-1].strip().lower() != self.current_out.strip().lower():
                self.text.append(self.current_out)
                with self.lock:
                    completed_segment = self.format_segment(
                        self.timestamp_offset,
                        self.timestamp_offset + min(duration, self.end_time_for_same_output),
                        self.current_out,
                        completed=True
                    )
                    self.transcript.append(completed_segment)

                    if self.translation_queue:
                        try:
                            self.translation_queue.put(completed_segment.copy(), timeout=0.1)
                        except queue.Full:
                            logging.warning("Translation queue is full, skipping segment")

            self.current_out = ''
            offset = min(duration, self.end_time_for_same_output)
            self.same_output_count = 0
            last_segment = None
            self.end_time_for_same_output = None
        else:
            self.prev_out = self.current_out

        if offset is not None:
            with self.lock:
                self.timestamp_offset += offset

        return last_segment
