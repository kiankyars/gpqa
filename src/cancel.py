import anthropic

client = anthropic.Anthropic()

for message_batch in client.messages.batches.list():
    if message_batch.processing_status == "in_progress":
        message_batch = client.messages.batches.cancel(
            message_batch.id,
        )

for message_batch in client.messages.batches.list():
    print(message_batch.processing_status, message_batch.id)