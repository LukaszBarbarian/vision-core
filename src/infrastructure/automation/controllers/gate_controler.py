# application/gate/gate_controller_worker.py

from __future__ import annotations

import asyncio
import traceback


class GateControllerWorker:
    def __init__(
        self,
        event_bus,
        gate_controller,
        command_topic: str = "gate.command.open",
    ):
        self.event_bus = event_bus
        self.gate_controller = gate_controller
        self.command_topic = command_topic

        self.running = False

    async def run(self):
        self.running = True
        queue = self.event_bus.subscribe(self.command_topic)

        print(
            f"[GateControllerWorker] Subscribed command topic: "
            f"{self.command_topic}"
        )

        try:
            while self.running:
                command = await queue.get()

                if not command:
                    continue

                await self._handle_command(command)

        except asyncio.CancelledError:
            pass

        except Exception:
            print("[GateControllerWorker] Worker crashed")
            traceback.print_exc()

        finally:
            self.event_bus.unsubscribe(self.command_topic, queue)
            print("[GateControllerWorker] Stopped")

    def stop(self):
        self.running = False

    async def _handle_command(self, command: dict):
        camera_id = command.get("camera_id")
        reason = command.get("reason")
        rule_name = command.get("rule_name")

        print(
            f"[GateControllerWorker] Opening gate. "
            f"camera={camera_id}, rule={rule_name}, reason={reason}"
        )


        result = None
        # result = self.gate_controller.open_gate(
        #     camera_id=camera_id,
        #     reason=reason,
        #     metadata=command,
        # )

        if asyncio.iscoroutine(result):
            await result