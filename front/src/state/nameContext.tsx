import React, { createContext, FC, useState } from "react";

export const NameContext = createContext({
  name: "",
  setName: (_any: any) => {},
});

export const NameProvider: FC = ({ children }) => {
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const [name, setName] = useState("");

  return (
    <NameContext.Provider
      value={{
        name,
        setName,
      }}
    >
      {children}
    </NameContext.Provider>
  );
};
