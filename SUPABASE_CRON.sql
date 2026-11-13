-- Run separately after enabling the Supabase Cron module in the dashboard.
-- One SQL cleanup run per hour. Database row expiry is still checked on read.
select cron.schedule(
    'hourly-rag-chat-expiry',
    '0 * * * *',
    $$delete from public.rag_chat_history where expires_at <= now()$$
);
