# 🚨 Google API Quota Exceeded - Solutions

## The Error You're Seeing

```
429 You exceeded your current quota
Quota exceeded for metric: generativelanguage.googleapis.com/embed_content_free_tier_requests
```

## What This Means

Google's Gemini API has **free tier limits**:
- ✅ Requests per minute: Limited
- ✅ Requests per day: Limited  
- ✅ You've hit one or more of these limits

## 🔧 Quick Solutions

### Solution 1: Wait It Out ⏰

**If you hit the per-minute limit:**
- Wait 1-2 minutes and try again

**If you hit the daily limit:**
- Wait 24 hours for quota reset
- Check your usage: https://ai.dev/usage?tab=rate-limit

### Solution 2: Use New Google API Key 🔑

1. Use a **different Google account**
2. Go to: https://makersuite.google.com/app/apikey
3. Create a new API key
4. Update `.env` with the new key:
   ```
   GOOGLE_API_KEY=your_new_key_here
   ```
5. Restart the app

### Solution 3: Upgrade to Paid Plan 💳

1. Visit: https://console.cloud.google.com/billing
2. Enable billing on your Google Cloud project
3. Get much higher quotas:
   - Free tier: ~1,500 requests/day
   - Paid tier: Much higher limits

Cost: Very cheap for embeddings (~$0.00025 per 1K tokens)

### Solution 4: Switch to Alternative Embeddings 🆓

**HuggingFace Embeddings** (Free & Unlimited)

If you want to avoid Google API limits entirely, you can use local embeddings:

#### Option A: HuggingFace Sentence Transformers
```bash
pip install sentence-transformers
```

Then modify the code to use:
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

#### Option B: OpenAI Embeddings
```bash
pip install langchain-openai
```

Requires OpenAI API key but has generous limits.

## 📊 Understanding Your Limits

**Free Tier Limits (Google Gemini):**
- ~1,500 requests per day
- ~15 requests per minute
- Per model, per project

**What Uses Quota:**
1. Creating vector store (ingestion.py) - Uses MANY requests
2. Each user question - Uses 1-2 requests

## ✅ Best Practices

1. **Create vector store ONCE**
   - Run `ingestion.py` only when needed
   - The `my_vector_store` folder is reusable

2. **Don't recreate unnecessarily**
   - If you already have `my_vector_store/`, don't run ingestion again
   - Just restart the app

3. **Use caching**
   - The app already loads from saved vector store
   - No need to regenerate embeddings each time

## 🔍 Check Your Usage

Monitor your quota usage:
- Visit: https://ai.dev/usage
- See how many requests you've made
- View when quota resets

## ⚠️ Common Mistakes

❌ **DON'T**: Run `ingestion.py` multiple times
- This uses up your quota very quickly
- Each PDF page = multiple embedding requests

✅ **DO**: Run `ingestion.py` once, save the vector store, reuse it

❌ **DON'T**: Delete `my_vector_store/` folder unless needed
- You'll have to regenerate everything

✅ **DO**: Keep the vector store and just run the app

## 🆘 Still Having Issues?

If none of these work:

1. **Check if vector store exists:**
   ```
   Legal-CHATBOT/my_vector_store/
   ```

2. **If it exists**, you don't need Google API for startup
   - The error might be from trying to process a query
   - Try with a fresh API key or wait for reset

3. **If it doesn't exist**, you need to create it
   - Use a different Google account with fresh quota
   - Or switch to HuggingFace embeddings

## 📝 Need to Switch Embedding Models?

Contact me if you want to switch to free unlimited HuggingFace embeddings. I can update both `app.py` and `ingestion.py` for you.
