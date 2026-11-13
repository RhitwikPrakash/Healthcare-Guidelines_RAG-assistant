-- Healthcare Guidelines RAG Assistant: 72-hour chat history.
-- Run once in Supabase Dashboard > SQL Editor, not in VS Code terminal.
-- Stores chat text, selected evidence snippets and references, NOT PDF bytes.

create table if not exists public.rag_chat_history (
    id uuid primary key,
    owner_id text not null,
    title text not null default 'New chat',
    messages jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    expires_at timestamptz not null default (now() + interval '72 hours')
);

create index if not exists rag_chat_history_owner_expiry_idx
    on public.rag_chat_history (owner_id, expires_at, updated_at desc);

-- Prevent access using public/anonymous Supabase API keys.
alter table public.rag_chat_history enable row level security;
revoke all on public.rag_chat_history from anon, authenticated;
grant select, insert, update, delete on public.rag_chat_history to service_role;

-- Refresh expiry only when a chat is actually saved (not when it is read).
create or replace function public.set_rag_chat_expiry()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
    new.updated_at := now();
    new.expires_at := now() + interval '72 hours';
    return new;
end;
$$;

drop trigger if exists refresh_rag_chat_expiry on public.rag_chat_history;
create trigger refresh_rag_chat_expiry
before insert or update on public.rag_chat_history
for each row execute function public.set_rag_chat_expiry();

