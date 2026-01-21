import anthropic, sys

client = anthropic.Anthropic()

if sys.argv[1] == "cancel":
    for message_batch in client.messages.batches.list():
        if message_batch.processing_status == "in_progress":
            message_batch = client.messages.batches.cancel(
                message_batch.id,
            )

if sys.argv[1] == "check":
    for message_batch in client.messages.batches.list():
        print(message_batch.processing_status, message_batch.id)

if sys.argv[1] == "content":
    for result in client.messages.batches.results(sys.argv[2]):
        print(result)