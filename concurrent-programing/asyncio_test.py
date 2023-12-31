import asyncio
import random

'''
set responce
'''
async def set_responce(queue):
    last_responce = 0

    while True:
        # Send get robot status command
        responce = random.randint(0, 100)

        if responce != last_responce:
            last_responce = responce
            print('set responce:' + str(responce))
            
        await queue.put(responce)
        await asyncio.sleep(1/500) # 500Hz = 1/500 = 0.002s = 2ms

'''
get responce
'''
async def get_responce(queue):
    while True:
        responce = await queue.get()
        print('get responce:' + str(responce))
        await asyncio.sleep(1/500) # 500Hz = 1/500 = 0.002s = 2ms

''' 
main function
'''
async def main():
    # Create a queue that we will use to store our "workload".
    queue = asyncio.Queue()

    # Create tasks for both functions to run concurrently
    task1 = asyncio.create_task(set_responce(queue))
    task2 = asyncio.create_task(get_responce(queue))

    # Wait for both tasks to complete
    await task1
    await task2


if __name__ == "__main__":
    # Start the main function
    asyncio.run(main())