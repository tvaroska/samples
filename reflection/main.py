"""

    Tests for agentic UI

"""
import os
from uuid import uuid4
import flet as ft

from langchain_core.messages import HumanMessage

from reflection import reflection

ROOT_DIR = '/tmp/flet'

async def main(page: ft.Page):

    page.title = 'Gemini Reflection'

    async def update_text(text):
        progress_text.value = text
        progress_text.update()

    async def update():
        await update_text('Generation no. 1')
        final = ''
        with open(page.session.get('file_name'), 'rb') as f:
            data = f.read()
        message = HumanMessage(
            content=[
                {'type': 'text', 'text': 'Create script for 5 minutes talk about the following paper. The audience for talk is undergraduate computer science students.'},
                            {
                                'type': 'media',
                                'mime_type': 'application/pdf',
                                'data': data,
                            },
                        ]
                    )
        async for chunk in agent.astream({'messages': [message]}, stream_mode='updates'):
            for _, values in chunk.items():
                if 'counter' in values:
                    counter = values['counter']
                    await update_text(f'Generation no.{values["counter"]+1}...')
                if 'final' in values:
                    if final != values['final']:
                        final = values['final']
                        tabs.tabs.append(ft.Tab(
                            text=f'Attempt {counter}',
                            content=ft.Column(
                                controls=[ft.Markdown(values['final'])]
                                )
                            )
                        )
                    tabs.update()
                    tabs.selected_index = counter
        progress_ring.value = 1
        upload_button.disabled = False
        page.update()

    async def on_upload_progress(e: ft.FilePickerUploadEvent):
        if e.progress == 1:
            e.page.run_task(update)

    async def get_file(e):
        upload_button.disabled = True
        progress_ring.visible = True
        progress_text.visible = True
        e.page.update()

        if e.files:
            progress_ring.visible = True
            progress_text.visible = True
            await update_text('Downloading')

            file = e.files[0]

            tmp_dir_name = str(uuid4())[:4]
            upload_file = ft.FilePickerUploadFile(
                file.name,
                upload_url=e.page.get_upload_url(f'{tmp_dir_name}/{file.name}', 60)
            )
            file_name = f'{ROOT_DIR}/{tmp_dir_name}/{file.name}'
            page.session.set('file_name', file_name)

            pick_file_dialog.upload([upload_file])


    # async def on_start(e: ft.ControlEvent):
    #     upload_button.disabled = True
    #     progress_ring.visible = True
    #     progress_text.visible = True
    #     e.page.update()
    #     e.page.run_task(update)

    agent = reflection()

    page.title = 'Gemini Reflection'
    pick_file_dialog = ft.FilePicker(on_result=get_file, on_upload=on_upload_progress)
    upload_button = ft.ElevatedButton(
                'Upload PDF',
                icon=ft.icons.START,
                on_click=lambda _: pick_file_dialog.pick_files(
                        allow_multiple=False
                                ))

    page.overlay.append(pick_file_dialog)

    progress_ring = ft.ProgressRing(height=20, width=20)
    progress_ring.visible = False

    progress_text = ft.Text('Starting to generate')
    progress_text.visible = False

    top_row = ft.Row(
        controls = [
            upload_button,
            progress_ring,
            progress_text
            ]
    )

    tabs = ft.Tabs(
        tabs = [],
#         visible=False,
        expand=True
    )

    bottom_row = ft.Row(
        expand=True,
        controls = [tabs]
    )

    page.add(
        top_row,
        ft.Divider(),
        bottom_row)

os.environ['FLET_SECRET_KEY'] = 'SECRET_KEY'
ft.app(main, upload_dir=ROOT_DIR, view=ft.AppView.WEB_BROWSER)
