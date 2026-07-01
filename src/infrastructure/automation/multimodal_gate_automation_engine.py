# application/automation/multimodal_gate_automation_engine.py

from __future__ import annotations

import asyncio
import re
import time
import traceback


class MultimodalGateAutomationEngine:
    def __init__(
        self,
        event_bus,
        camera_config,
        default_max_event_age_seconds: float = 8.0,
        default_cooldown_seconds: float = 20.0,
    ):
        self.event_bus = event_bus
        self.config = camera_config

        self.default_max_event_age_seconds = float(default_max_event_age_seconds)
        self.default_cooldown_seconds = float(default_cooldown_seconds)

        self.running = False

        self.last_tripwire_by_camera: dict[str, dict] = {}
        self.last_speech_by_camera: dict[str, dict] = {}
        self.last_triggered_by_rule: dict[str, float] = {}

    async def start_processing(self):
        self.running = True

        multimodal_rules = self._get_multimodal_rules()

        if not multimodal_rules:
            print(
                f"[MultimodalGateAutomation] No multimodal rules for "
                f"{self.config.camera.id}"
            )
            return

        visual_topics = self._get_visual_topics(multimodal_rules)
        speech_topic = "speech.transcribed"

        subscriptions: list[tuple[str, object]] = []

        for topic in visual_topics:
            subscriptions.append((topic, self.event_bus.subscribe(topic)))

        subscriptions.append((speech_topic, self.event_bus.subscribe(speech_topic)))

        print(
            f"[MultimodalGateAutomation] {self.config.camera.id}: "
            f"subscribed visual={visual_topics}, speech={speech_topic}"
        )

        try:
            await self._event_loop(subscriptions, multimodal_rules)

        except asyncio.CancelledError:
            pass

        except Exception:
            print(
                f"[MultimodalGateAutomation] Worker crashed for "
                f"{self.config.camera.id}"
            )
            traceback.print_exc()

        finally:
            for topic, queue in subscriptions:
                self.event_bus.unsubscribe(topic, queue)

            print(
                f"[MultimodalGateAutomation] Stopped for "
                f"{self.config.camera.id}"
            )

    def stop(self):
        self.running = False

    async def _event_loop(
        self,
        subscriptions: list[tuple[str, object]],
        multimodal_rules: list[dict],
    ):
        while self.running:
            event = await self._wait_for_any_event(subscriptions)

            if not event:
                continue

            event_camera_id = event.get("camera_id")

            if event_camera_id != self.config.camera.id:
                continue

            event_type = event.get("type")

            if event_type == "tripwire.crossed":
                self.last_tripwire_by_camera[event_camera_id] = event

            elif event_type == "speech.transcribed":
                self.last_speech_by_camera[event_camera_id] = event

            else:
                continue

            for rule in multimodal_rules:
                await self._try_trigger_rule(rule, event_camera_id)

    async def _try_trigger_rule(
        self,
        rule: dict,
        camera_id: str,
    ):
        now = time.time()

        tripwire = self.last_tripwire_by_camera.get(camera_id)
        speech = self.last_speech_by_camera.get(camera_id)

        if not tripwire or not speech:
            return

        visual_event = rule.get("visual_event")

        if visual_event and tripwire.get("trigger_event") != visual_event:
            return

        max_event_age_seconds = float(
            rule.get("max_event_age_seconds", self.default_max_event_age_seconds)
        )

        tripwire_ts = float(tripwire.get("timestamp") or 0.0)
        speech_ts = float(
            speech.get("timestamp_end")
            or speech.get("timestamp")
            or 0.0
        )

        if now - tripwire_ts > max_event_age_seconds:
            return

        if now - speech_ts > max_event_age_seconds:
            return

        speech_text = str(speech.get("text") or "")
        normalized_text = self._normalize_text(speech_text)

        if not self._matches_speech_rule(normalized_text, rule):
            return

        rule_name = rule.get("name", "multimodal_gate_open")
        cooldown_key = f"{camera_id}:{rule_name}"

        cooldown_seconds = float(
            rule.get("cooldown_seconds", self.default_cooldown_seconds)
        )

        last_triggered = self.last_triggered_by_rule.get(cooldown_key, 0.0)

        if now - last_triggered <= cooldown_seconds:
            return

        self.last_triggered_by_rule[cooldown_key] = now

        command_event = rule.get("command_event", "gate.command.open")

        command = {
            "type": command_event,
            "camera_id": camera_id,
            "rule_name": rule_name,
            "reason": "tripwire_and_speech",
            "speech_text": speech_text,
            "normalized_speech_text": normalized_text,
            "timestamp": now,
            "tripwire_event": tripwire,
            "speech_event": speech,
        }

        print(
            f"[MultimodalGateAutomation] 🔓 Gate command: "
            f"{command_event}, camera={camera_id}, "
            f"rule={rule_name}, speech={speech_text!r}"
        )

        await self.event_bus.publish(command_event, command)

    def _get_multimodal_rules(self) -> list[dict]:
        rules = getattr(self.config.camera, "rules", []) or []

        return [
            rule
            for rule in rules
            if rule.get("type") == "multimodal_gate_open"
        ]

    def _get_visual_topics(self, multimodal_rules: list[dict]) -> list[str]:
        topics = []

        for rule in multimodal_rules:
            topic = rule.get("visual_event")

            if topic and topic not in topics:
                topics.append(topic)

        return topics

    def _matches_speech_rule(
        self,
        normalized_text: str,
        rule: dict,
    ) -> bool:
        phrases = rule.get("speech_phrases") or []

        normalized_phrases = [
            self._normalize_text(phrase)
            for phrase in phrases
            if phrase
        ]

        if not normalized_phrases:
            return False

        return any(
            phrase in normalized_text
            for phrase in normalized_phrases
        )

    def _normalize_text(self, text: str) -> str:
        text = text.lower().strip()

        replacements = {
            "ą": "a",
            "ć": "c",
            "ę": "e",
            "ł": "l",
            "ń": "n",
            "ó": "o",
            "ś": "s",
            "ź": "z",
            "ż": "z",
        }

        for source, target in replacements.items():
            text = text.replace(source, target)

        text = re.sub(r"[^a-z0-9 ]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    async def _wait_for_any_event(
        self,
        subscriptions: list[tuple[str, object]],
    ) -> dict | None:
        tasks = [
            asyncio.create_task(queue.get())
            for _, queue in subscriptions
        ]

        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        for task in done:
            return task.result()

        return None