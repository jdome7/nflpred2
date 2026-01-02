import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("https://nflpred.streamlit.app/")
        
        wake_button = page.get_by_role("button", name="Yes, get this app back up")
        if await wake_button.is_visible():
            print("App is asleep. Clicking wake up button...")
            await wake_button.click()
            
            await page.wait_for_timeout(10000) 
        else:
            print("App is already awake!")
            
        await browser.close()

asyncio.run(run())