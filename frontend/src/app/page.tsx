'use client';

import { useState } from 'react';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async () => {
    if (!question.trim()) return;
    
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        setAnswer(data.answer);
      } else {
        setAnswer('Sorry, I encountered an error while processing your question.');
      }
    } catch (error) {
      setAnswer('Error: Unable to connect to the backend. Make sure the server is running on localhost:8000.');
    }
    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="relative flex size-full min-h-screen flex-col bg-white group/design-root overflow-x-hidden">
      <div className="layout-container flex h-full grow flex-col">
        {/* Header */}
        <header className="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#f0f4f0] px-10 py-3">
          <div className="flex items-center gap-4 text-[#111811]">
            <div className="size-4">
              <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path
                  fillRule="evenodd"
                  clipRule="evenodd"
                  d="M39.475 21.6262C40.358 21.4363 40.6863 21.5589 40.7581 21.5934C40.7876 21.655 40.8547 21.857 40.8082 22.3336C40.7408 23.0255 40.4502 24.0046 39.8572 25.2301C38.6799 27.6631 36.5085 30.6631 33.5858 33.5858C30.6631 36.5085 27.6632 38.6799 25.2301 39.8572C24.0046 40.4502 23.0255 40.7407 22.3336 40.8082C21.8571 40.8547 21.6551 40.7875 21.5934 40.7581C21.5589 40.6863 21.4363 40.358 21.6262 39.475C21.8562 38.4054 22.4689 36.9657 23.5038 35.2817C24.7575 33.2417 26.5497 30.9744 28.7621 28.762C30.9744 26.5497 33.2417 24.7574 35.2817 23.5037C36.9657 22.4689 38.4054 21.8562 39.475 21.6262ZM4.41189 29.2403L18.7597 43.5881C19.8813 44.7097 21.4027 44.9179 22.7217 44.7893C24.0585 44.659 25.5148 44.1631 26.9723 43.4579C29.9052 42.0387 33.2618 39.5667 36.4142 36.4142C39.5667 33.2618 42.0387 29.9052 43.4579 26.9723C44.1631 25.5148 44.659 24.0585 44.7893 22.7217C44.9179 21.4027 44.7097 19.8813 43.5881 18.7597L29.2403 4.41187C27.8527 3.02428 25.8765 3.02573 24.2861 3.36776C22.6081 3.72863 20.7334 4.58419 18.8396 5.74801C16.4978 7.18716 13.9881 9.18353 11.5858 11.5858C9.18354 13.988 7.18717 16.4978 5.74802 18.8396C4.58421 20.7334 3.72865 22.6081 3.36778 24.2861C3.02574 25.8765 3.02429 27.8527 4.41189 29.2403Z"
                  fill="currentColor"
                />
              </svg>
            </div>
            <h2 className="text-[#111811] text-lg font-bold leading-tight tracking-[-0.015em]">Zen Note</h2>
          </div>
          <div className="flex flex-1 justify-end gap-8">
            <div className="flex items-center gap-9">
              <a className="text-[#111811] text-sm font-medium leading-normal" href="#">Home</a>
              <a className="text-[#111811] text-sm font-medium leading-normal" href="#">About</a>
              <a className="text-[#111811] text-sm font-medium leading-normal" href="#">Contact</a>
            </div>
            <button
              className="flex max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 bg-[#f0f4f0] text-[#111811] gap-2 text-sm font-bold leading-normal tracking-[0.015em] min-w-0 px-2.5"
              aria-label="Help"
              tabIndex={0}
            >
              <div className="text-[#111811]">
                <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" fill="currentColor" viewBox="0 0 256 256">
                  <path
                    d="M140,180a12,12,0,1,1-12-12A12,12,0,0,1,140,180ZM128,72c-22.06,0-40,16.15-40,36v4a8,8,0,0,0,16,0v-4c0-11,10.77-20,24-20s24,9,24,20-10.77,20-24,20a8,8,0,0,0-8,8v8a8,8,0,0,0,16,0v-.72c18.24-3.35,32-17.9,32-35.28C168,88.15,150.06,72,128,72Zm104,56A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"
                  />
                </svg>
              </div>
            </button>
          </div>
        </header>

        {/* Main Content */}
        <div className="px-40 flex flex-1 justify-center py-5">
          <div className="layout-content-container flex flex-col max-w-[960px] flex-1">
            <h2 className="text-[#111811] tracking-light text-[28px] font-bold leading-tight px-4 text-center pb-3 pt-5">
              Ask me anything about your notes
            </h2>
            
            {/* Question Input */}
            <div className="flex max-w-[480px] flex-wrap items-end gap-4 px-4 py-3">
              <label className="flex flex-col min-w-40 flex-1">
                <input
                  placeholder="Type your question here"
                  className="form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-xl text-[#111811] focus:outline-0 focus:ring-0 border border-[#dbe6db] bg-white focus:border-[#dbe6db] h-14 placeholder:text-[#618961] p-[15px] text-base font-normal leading-normal"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </label>
            </div>
            
            {/* Submit Button */}
            <div className="flex px-4 py-3 justify-center">
              <button
                className="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-full h-10 px-4 bg-[#075907] text-white text-sm font-bold leading-normal tracking-[0.015em] disabled:opacity-50 disabled:cursor-not-allowed"
                onClick={handleSubmit}
                disabled={isLoading || !question.trim()}
                aria-label="Submit question"
                tabIndex={0}
              >
                <span className="truncate">
                  {isLoading ? 'Processing...' : 'Submit'}
                </span>
              </button>
            </div>
            
            {/* Answer Area */}
            <div className="px-4 py-3">
              {answer ? (
                <div className="bg-[#f8f9fa] border border-[#dbe6db] rounded-xl p-4">
                  <h3 className="text-[#111811] text-lg font-semibold mb-2">Answer:</h3>
                  <p className="text-[#111811] text-base font-normal leading-relaxed whitespace-pre-wrap">
                    {answer}
                  </p>
                </div>
              ) : (
                <p className="text-[#618961] text-base font-normal leading-normal pb-3 pt-1">
                  Answer will appear here
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
