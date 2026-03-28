"""FSM-состояния Telegram-бота."""
from aiogram.fsm.state import State, StatesGroup


class LinkStates(StatesGroup):
    waiting_code = State()


class MasterStates(StatesGroup):
    waiting_file = State()
    choosing_preset = State()


class AnalyzeStates(StatesGroup):
    waiting_file = State()
