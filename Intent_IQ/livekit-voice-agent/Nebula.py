from dotenv import load_dotenv
from livekit.agents import RoomInputOptions
from livekit import agents
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from tool import CallTools
from prompt import AGENT_INSTRUCTIONS
import logging
logger = logging.getLogger("Nebula.py")
load_dotenv(".env.local")
class Assistant(agents.Agent):
    def __init__(self):
        self.my_tools = CallTools(agent=self)
        super().__init__(
            instructions=AGENT_INSTRUCTIONS,
            tools=[self.my_tools.end_call, self.my_tools.transfer_to_human])
async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    logger.info(f"Phone call connected from participant: {participant.identity}")
    session = agents.AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(
                room=ctx.room,
                agent=Assistant(),
                room_input_options=RoomInputOptions(
                    # For telephony applications, use `BVCTelephony` instead for best results
                    noise_cancellation=noise_cancellation.BVC(),
                ),
            )
    await session.generate_reply(
            instructions="Greet the user and offer your assistance."
        )
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint,
        agent_name="Nebula")
    )
